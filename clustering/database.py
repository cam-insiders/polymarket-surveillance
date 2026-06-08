from collections import defaultdict
import logging
import time
from typing import Dict, List, Optional, Set, Tuple
from clustering.models import ClusterInfo, EntryRecord
import sqlite3
import queue

class ClusteringDatabase:
    """Manages SQLite persistence for clustering state"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        logging.info(f"Clustering database initialized: {db_path}")
    
    def _init_schema(self):
        """
        Create tables if they don't exist.
        """
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='market_participation'"
        )
        table_exists = cursor.fetchone() is not None

        if table_exists:
            # Check if the unique constraint already exists by inspecting
            # the CREATE TABLE statement stored in sqlite_master.
            cursor = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='market_participation'"
            )
            create_sql = cursor.fetchone()[0] or ""
            needs_migration = "UNIQUE" not in create_sql.upper()

            if needs_migration:
                logging.info(
                    "Migrating market_participation to add UNIQUE constraint "
                    "(this deduplicates existing rows)"
                )
                self.conn.executescript("""
                    CREATE TABLE IF NOT EXISTS market_participation_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        market_id TEXT NOT NULL,
                        wallet TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        direction TEXT NOT NULL,
                        size REAL NOT NULL,
                        outcome_index INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(market_id, wallet, timestamp, direction, outcome_index)
                    );

                    INSERT OR IGNORE INTO market_participation_new
                        (market_id, wallet, timestamp, direction, size, outcome_index, created_at)
                    SELECT market_id, wallet, timestamp, direction, size, outcome_index, created_at
                    FROM market_participation;

                    DROP TABLE market_participation;
                    ALTER TABLE market_participation_new RENAME TO market_participation;
                """)
                self.conn.commit()

                old_count_cursor = self.conn.execute("SELECT COUNT(*) FROM market_participation")
                new_count = old_count_cursor.fetchone()[0]
                logging.info(f"Migration complete: {new_count} deduplicated entries retained")

        self.conn.executescript("""
            -- Raw entry data (append-only, deduplicated on natural key)
            CREATE TABLE IF NOT EXISTS market_participation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                wallet TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                direction TEXT NOT NULL,
                size REAL NOT NULL,
                outcome_index INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(market_id, wallet, timestamp, direction, outcome_index)
            );
            
            CREATE INDEX IF NOT EXISTS idx_market_wallet 
                ON market_participation(market_id, wallet);
            CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON market_participation(timestamp);
            
            -- Co-activity edges (retained for debugging; not used by bucket builder)
            CREATE TABLE IF NOT EXISTS coactivity_edges (
                wallet_a TEXT NOT NULL,
                wallet_b TEXT NOT NULL,
                weight REAL NOT NULL,
                last_updated INTEGER NOT NULL,
                PRIMARY KEY (wallet_a, wallet_b),
                CHECK (wallet_a < wallet_b)
            );
            
            CREATE INDEX IF NOT EXISTS idx_edge_weight 
                ON coactivity_edges(weight);
            
            -- Cluster assignments (computed periodically)
            CREATE TABLE IF NOT EXISTS cluster_assignments (
                wallet TEXT PRIMARY KEY,
                cluster_id INTEGER,
                computed_at INTEGER NOT NULL
            );
            
            CREATE INDEX IF NOT EXISTS idx_cluster_id 
                ON cluster_assignments(cluster_id);
            
            -- Cluster metadata (computed periodically)
            CREATE TABLE IF NOT EXISTS cluster_metadata (
                cluster_id INTEGER PRIMARY KEY,
                size INTEGER NOT NULL,
                density REAL NOT NULL,
                total_edge_weight REAL NOT NULL,
                has_common_ownership INTEGER DEFAULT 0,
                attribution_enriched INTEGER DEFAULT 0,
                computed_at INTEGER NOT NULL
            );
            
            -- Attribution edges (Layer 2, populated lazily)
            CREATE TABLE IF NOT EXISTS attribution_edges (
                from_wallet TEXT NOT NULL,
                to_wallet TEXT NOT NULL,
                total_amount REAL NOT NULL,
                tx_count INTEGER NOT NULL,
                first_tx INTEGER NOT NULL,
                last_tx INTEGER NOT NULL,
                tx_hashes TEXT,
                PRIMARY KEY (from_wallet, to_wallet)
            );
            
            -- Attribution query cache (prevent redundant API calls)
            CREATE TABLE IF NOT EXISTS attribution_cache (
                wallet TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                queried_at INTEGER NOT NULL
            );
        """)
        self.conn.commit()

    def write_entry(self, market_id: str, wallet: str, entry: EntryRecord):
        """
        Write a single entry to the database.
        """
        try:
            self.conn.execute("""
            INSERT OR IGNORE INTO market_participation (
                market_id, wallet, timestamp, direction, size, outcome_index
            ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                market_id,
                wallet,
                entry.timestamp,
                entry.direction,
                entry.size,
                entry.outcome_index
            ))
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            logging.warning(f"Failed to write entry: {e}")
            self.conn.rollback()
    
    def write_edge_batch(self, edges: List[Tuple[str, str, float, int]]):
        """
        Write or update multiple edges atomically.
        
        Note: Not used by the bucket projection manager (edges are derived
        from entries on each rebuild). Retained for debugging / migration.
        """
        if not edges:
            return
        
        try:
            self.conn.executemany("""
            INSERT OR REPLACE INTO coactivity_edges (
                wallet_a, wallet_b, weight, last_updated
            ) VALUES (?, ?, ?, ?)
            """, edges)
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            logging.warning(f"Failed to write edge batch: {e}")
            self.conn.rollback()
    
    def write_cluster_state(
        self,
        wallet_to_cluster: Dict[str, Optional[int]],
        cluster_metadata: Dict[int, ClusterInfo],
        computed_at: int
    ):
        """Write complete cluster state atomically"""
        try:
            self.conn.execute("DELETE FROM cluster_assignments")
            self.conn.execute("DELETE FROM cluster_metadata")

            self.conn.executemany("""
                INSERT INTO cluster_assignments (wallet, cluster_id, computed_at)
                VALUES (?, ?, ?)
                """, [(wallet, cluster_id, computed_at)
                for wallet, cluster_id in wallet_to_cluster.items()
            ])

            self.conn.executemany("""
                INSERT INTO cluster_metadata 
                    (cluster_id, size, density, total_edge_weight, 
                    has_common_ownership, attribution_enriched, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    info.cluster_id, info.size, info.density,
                    info.total_edge_weight, 
                    1 if info.has_common_ownership else 0,
                    1 if info.attribution_enriched else 0,
                    computed_at
                )
                for info in cluster_metadata.values()
            ])
            self.conn.commit()
            logging.info(f"Wrote cluster state: {len(cluster_metadata)} clusters")
        except sqlite3.IntegrityError as e:
            logging.warning(f"Failed to write cluster state: {e}")
            self.conn.rollback()
    
    def load_all_entries(self) -> Dict[str, Dict[str, List[EntryRecord]]]:
        """Load all market entries from the database for graph reconstruction"""
        
        entries = defaultdict(lambda: defaultdict(list))
        cursor = self.conn.execute("""
            SELECT market_id, wallet, timestamp, direction, size, outcome_index
            FROM market_participation
            ORDER BY timestamp ASC
        """)

        for row in cursor:
            entry = EntryRecord(
                timestamp=row["timestamp"],
                direction=row["direction"],
                size=row["size"],
                outcome_index=row["outcome_index"]
            )
            entries[row["market_id"]][row["wallet"]].append(entry)
        
        logging.info(f"Loaded {sum(len(w) for m in entries.values() for w in m.values())} entries")
        return dict(entries)
    
    def load_cluster_assignments(self) -> Dict[str, Optional[int]]:
        """Load all cluster assignments from the database"""
        cursor = self.conn.execute("""
            SELECT wallet, cluster_id
            FROM cluster_assignments
        """)
        return {row["wallet"]: row["cluster_id"] for row in cursor}
    
    def load_edges(self) -> List[Tuple[str, str, float]]:
        """
        Load all co-activity edges from the database.
        
        Note: Not used by the bucket projection manager (edges are
        derived from entries). Retained for debugging / inspection.
        """
        cursor = self.conn.execute("""
            SELECT wallet_a, wallet_b, weight
            FROM coactivity_edges
            ORDER BY weight DESC
        """)

        edges = [(row["wallet_a"], row["wallet_b"], row["weight"]) for row in cursor]
        logging.info(f"Loaded {len(edges)} edges from database")
        return edges

    def load_cluster_metadata(self) -> Dict[int, Dict]:
        """Load cluster metadata from the database"""

        cursor = self.conn.execute("""
            SELECT cluster_id, size, density, total_edge_weight,
                has_common_ownership, attribution_enriched, computed_at
            FROM cluster_metadata
        """)
        metadata = {}
        for row in cursor:
            metadata[row["cluster_id"]] = {
                "size": row["size"],
                "density": row["density"],
                "total_edge_weight": row["total_edge_weight"],
                "has_common_ownership": bool(row["has_common_ownership"]),
                "attribution_enriched": bool(row["attribution_enriched"]),
                "computed_at": row["computed_at"],
                "wallets": set()
            }
        
        cursor = self.conn.execute("""
            SELECT wallet, cluster_id
            FROM cluster_assignments
            WHERE cluster_id IS NOT NULL
        """)

        for row in cursor:
            cluster_id = row["cluster_id"]
            if cluster_id in metadata:
                metadata[cluster_id]["wallets"].add(row["wallet"])
            else:
                logging.warning(
                    f"Wallet {row['wallet']} assigned to unknown cluster {cluster_id}"
                )
        
        for cluster_id, data in metadata.items():
            expected_size = data['size']
            actual_size = len(data['wallets'])
            if expected_size != actual_size:
                logging.warning(
                    f"Cluster {cluster_id} size mismatch: "
                    f"metadata says {expected_size}, but found {actual_size} wallets"
                )
        
        logging.info(f"Loaded metadata for {len(metadata)} clusters")
        return metadata

    def validate_edges_consistency(
            self,
            entries: Dict[str, Dict[str, List[EntryRecord]]],
            loaded_edges: List[Tuple[str, str, float]]
    ) -> bool:
        """
        Validate that loaded edges are consistent with entries.
        
        Note: Not used by the bucket projection manager. Retained for
        potential debugging of legacy data.
        """
        wallets_in_entries = set()
        for market_entries in entries.values():
            wallets_in_entries.update(market_entries.keys())
        
        invalid_edges = 0
        for wallet_a, wallet_b, weight in loaded_edges:
            if wallet_a not in wallets_in_entries or wallet_b not in wallets_in_entries:
                invalid_edges += 1
                if invalid_edges <= 5:
                    logging.warning(
                        f"Edge references unknown wallet: {wallet_a[:10]}.../{wallet_b[:10]}..."
                    )
        if invalid_edges > 0:
            logging.warning(f"Total invalid edges: {invalid_edges}")
            return False

        wallet_count = len(wallets_in_entries)
        max_possible_edges = wallet_count * (wallet_count - 1) // 2

        if len(loaded_edges) > max_possible_edges:
            logging.warning(
                f"Loaded edges exceed maximum possible: "
                f"{len(loaded_edges)} > {max_possible_edges}"
            )
            return False
        
        logging.info("Edge consistency validation passed")
        return True
    
    # Layer 2

    def get_unqueried_wallets(self, wallets: List[str]) -> List[str]:
        """Filter list of wallets to only those not in attribution cache"""

        if not wallets:
            return []
        
        placeholders = ",".join("?" * len(wallets))
        cursor = self.conn.execute(f"""
            SELECT wallet FROM attribution_cache
            WHERE wallet IN ({placeholders})
        """, wallets)

        queried_wallets = {row["wallet"] for row in cursor}
        unqueried = [w for w in wallets if w not in queried_wallets]

        logging.debug(
            f"Attribution cache: {len(queried_wallets)} already queried, "
            f"{len(unqueried)} need queries"
        )
        return unqueried
    
    def update_attribution_cache(self, wallet: str, status: str):
        """Mark a wallet as queried in the attribution cache"""

        try:
            self.conn.execute("""
            INSERT OR REPLACE INTO attribution_cache (
                wallet, status, queried_at
            ) VALUES (?, ?, ?)
            """, (wallet, status, int(time.time())))
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            logging.warning(f"Failed to update attribution cache for {wallet[:10]}...: {e}")
            self.conn.rollback()
    
    def write_attribution_edges(self, edges: List[Tuple]):
        """Write USDC transfer edges to the database"""
        if not edges:
            return
        
        try:
            self.conn.executemany("""
                INSERT OR REPLACE INTO attribution_edges (
                    from_wallet, to_wallet, total_amount, tx_count,
                    first_tx, last_tx, tx_hashes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, edges)
            self.conn.commit()
            logging.info(f"Wrote {len(edges)} attribution edges to database")
        except sqlite3.IntegrityError as e:
            logging.warning(f"Failed to write attribution edges: {e}")
            self.conn.rollback()
    
    def get_attribution_edges_for_cluster(self, wallets: Set[str]) -> List[Tuple]:
        """Fetch all attribution edges between wallets in a cluster"""
        if not wallets:
            return []
        
        placeholders = ",".join("?" * len(wallets))
        cursor = self.conn.execute(f"""
            SELECT from_wallet, to_wallet, total_amount, tx_count
            FROM attribution_edges
            WHERE from_wallet IN ({placeholders})
              AND to_wallet IN ({placeholders})
        """, list(wallets) * 2)

        edges = [
            (row["from_wallet"], row["to_wallet"], row["total_amount"], row["tx_count"])
            for row in cursor
        ]
        
        logging.debug(f"Found {len(edges)} attribution edges for cluster")
        return edges  

    def close(self):
        """Close the database connection"""
        self.conn.close()