from __future__ import annotations

import dataclasses
import logging
import sqlite3
import sys
from typing import Any, Final

from qualia_core.experimenttracking.QualiaDatabase import QualiaDatabase as QualiaDatabaseQualiaCore
from qualia_core.typing import TYPE_CHECKING

from qualia_plugin_snn.learningmodel.pytorch.SNN import SNN

if TYPE_CHECKING:
    from qualia_core.qualia import TrainResult

    from qualia_plugin_snn.postprocessing.OperationCounter import OperationMetrics

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class QualiaDatabase(QualiaDatabaseQualiaCore):
    # Latest schema extension to create fresh tables
    __sql_schema_snn: Final[str] = """
    CREATE TABLE IF NOT EXISTS models_snn (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id INTEGER,
        is_snn INTEGER,
        timesteps INTEGER,

        UNIQUE(model_id),
        FOREIGN KEY(model_id) REFERENCES models(id)
    );

    CREATE TABLE IF NOT EXISTS models_operationcounter (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id INTEGER,
        syn_acc REAL,
        syn_mac REAL,
        addr_acc REAL,
        addr_mac REAL,
        total_acc REAL,
        total_mac REAL,
        mem_write REAL,
        mem_read REAL,
        input_spikerate REAL,
        output_spikerate REAL,
        input_count REAL,
        output_count REAL,
        input_is_binary INTEGER,
        output_is_binary INTEGER,

        UNIQUE(model_id),
        FOREIGN KEY(model_id) REFERENCES models(id)
    );
    """

    # Incremental schema extension upgrades
    __sql_schema_upgrades_snn: Final[list[str]] = [
        """
        ALTER TABLE models_snn ADD COLUMN is_snn INTEGER;
        """,
        """
        CREATE TABLE IF NOT EXISTS models_operationcounter (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER,
            syn_acc REAL,
            syn_mac REAL,
            addr_acc REAL,
            addr_mac REAL,
            mem_write REAL,
            mem_read REAL,
            input_spikerate REAL,
            output_spikerate REAL,
            input_count REAL,
            output_count REAL,
            input_is_binary INTEGER,
            output_is_binary INTEGER,

            UNIQUE(model_id),
            FOREIGN KEY(model_id) REFERENCES models(id)
        );
        """,
        """
        ALTER TABLE models_operationcounter ADD COLUMN total_acc REAL;
        """,
        """
        ALTER TABLE models_operationcounter ADD COLUMN total_mac REAL;
        """,
    ]

    __queries_snn: Final[dict[str, str]] = {
        'get_schema_version_snn': "SELECT schema_version FROM plugins WHERE name = 'qualia_plugin_snn'",
        'set_schema_version_snn': "INSERT OR REPLACE INTO plugins(name, schema_version) VALUES ('qualia_plugin_snn', :version)",
        'insert_model_snn': 'INSERT OR REPLACE INTO models_snn(model_id, timesteps) VALUES(:model_id, :timesteps)',
        'insert_model_operationcounter': """INSERT OR REPLACE INTO models_operationcounter(
            model_id,
            syn_acc,
            syn_mac,
            addr_acc,
            addr_mac,
            total_acc,
            total_mac,
            mem_write,
            mem_read,
            input_spikerate,
            output_spikerate,
            input_count,
            output_count,
            input_is_binary,
            output_is_binary
        ) VALUES (
            :model_id,
            :syn_acc,
            :syn_mac,
            :addr_acc,
            :addr_mac,
            :total_acc,
            :total_mac,
            :mem_write,
            :mem_read,
            :input_spikerate,
            :output_spikerate,
            :input_count,
            :output_count,
            :input_is_binary,
            :output_is_binary
        )""",
        'get_model_snn': 'SELECT * from models_snn WHERE model_id = :model_id',
        'get_model_operationcounter': 'SELECT * from models_operationcounter WHERE model_id = :model_id',
    }

    def __set_schema_version_snn(self, cur: sqlite3.Cursor, version: int) -> None:
        _ = cur.execute(self.__queries_snn['set_schema_version_snn'], {'version': version})

    def __get_schema_version_snn(self, cur: sqlite3.Cursor) -> int | None:
        res = cur.execute(self.__queries_snn['get_schema_version_snn']).fetchone()
        return res[0] if res is not None else None

    def __upgrade_database_schema_snn(self, con: sqlite3.Connection, cur: sqlite3.Cursor) -> None:
        current_version = self.__get_schema_version_snn(cur)
        latest_version = self.__sql_schema_version_snn
        logger.info('Current database SNN schema extension version: %s, latest schema version: %d',
                    current_version, latest_version)

        # Initialize schema extension with fresh tables
        if current_version is None:
            _ = cur.execute('BEGIN')  # Begin transaction to only update version number if schema upgrade succeeded
            _ = cur.executescript(self.__sql_schema_snn)
            self.__set_schema_version_snn(cur, self.__sql_schema_version_snn)
            con.commit()
            return

        # Upgrade existing tables
        for i, sql_schema_upgrade in enumerate(self.__sql_schema_upgrades_snn[current_version:latest_version]):
            new_version = current_version + i + 1
            logger.info('Upgrading database schema to version %d', new_version)
            try:
                _ = cur.execute('BEGIN')  # Begin transaction to only update version number if schema upgrade succeeded
                _ = cur.execute(sql_schema_upgrade)
                self.__set_schema_version_snn(cur, new_version)
                con.commit()
            except sqlite3.Error:
                con.rollback()
                logger.exception('Could not upgrade database schema to version %d', new_version)

    @override
    def _upgrade_database_schema(self, con: sqlite3.Connection, cur: sqlite3.Cursor) -> None:
        super()._upgrade_database_schema(con, cur)
        self.__upgrade_database_schema_snn(con, cur)

    def __get_model_snn(self, cur: sqlite3.Cursor, model_id: int) -> dict[str, Any] | None:
        res = cur.execute(self.__queries_snn['get_model_snn'], {'model_id': model_id}).fetchone()
        return res if res is not None else None

    def __get_model_operationcounter(self, cur: sqlite3.Cursor, model_id: int) -> dict[str, Any] | None:
        res = cur.execute(self.__queries_snn['get_model_operationcounter'], {'model_id': model_id}).fetchone()
        return res if res is not None else None

    @override
    def log_trainresult(self, trainresult: TrainResult) -> int | None:
        model_id = super().log_trainresult(trainresult)

        if not self._con or not self._cur:
            logger.error('Database not initialized')
            return None

        snn_metadata = {
            'model_id': model_id,
            'is_snn': isinstance(trainresult.model, SNN) or getattr(trainresult.model, 'is_snn', False),
            'timesteps': getattr(trainresult.model, 'timesteps', 1),
        }

        _ = self._cur.execute(self.__queries_snn['insert_model_snn'], snn_metadata)
        self._con.commit()

    def log_operationcounter(self, model_hash: str, oms: list[OperationMetrics]) -> None:
        if not self._con or not self._cur:
            logger.error('Database not initialized')
            return

        model_id = self._lookup_model_hash(self._cur, model_hash)

        if model_id is None:
            logger.error('Model hash %s not found', model_hash)
            return

        om_total = next((om for om in oms if om.name == 'Total'), None)

        if not om_total:
            logger.error('Could not find Total in OperationMetrics')
            return

        operationcounter_data = {'model_id': model_id, **om_total.asdict()}

        _ = self._cur.execute(self.__queries_snn['insert_model_operationcounter'], operationcounter_data)
        self._con.commit()

    def __print_model_operationcounter(self, operationcounter: dict[str, Any]) -> None:
        max_name_length = max(len(k) for k in operationcounter)

        operationcounter.pop('id')
        operationcounter.pop('model_id')

        print('Operation counter:')
        for k, v in dict(operationcounter).items():
            print(f'    {k}: {" " * (max_name_length - len(k))}{v}')

    @override
    def _print_model(self, model: dict[str, Any]) -> None:
        super()._print_model(model)

        if not self._cur:
            logger.error('Database not initialized')
            return

        model_id = model['id']

        model_snn = self.__get_model_snn(self._cur, model_id=model_id)

        if model_snn:
            is_snn = bool(model_snn['is_snn'])
            print(f'SNN:              {is_snn}')
            print(f'Timesteps:        {model_snn["timesteps"]}')

        model_operationcounter = self.__get_model_operationcounter(self._cur, model_id=model_id)
        if model_operationcounter:
            self.__print_model_operationcounter(dict(model_operationcounter))

    @property
    def __sql_schema_version_snn(self) -> int:
        return len(self.__sql_schema_upgrades_snn)
