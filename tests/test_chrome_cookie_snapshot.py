import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.chrome_cookie_snapshot import (
    primary_cookie_database,
    yt_dlp_cookie_session,
)


def create_cookie_database(path: Path, hosts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE cookies (host_key TEXT)")
        connection.executemany(
            "INSERT INTO cookies (host_key) VALUES (?)",
            [(host,) for host in hosts],
        )
        connection.commit()
    finally:
        connection.close()


class ChromeCookieSnapshotTest(unittest.TestCase):
    def test_primary_database_ignores_nested_extension_cookie_store(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "Default"
            primary = profile / "Cookies"
            nested = profile / "Storage" / "ext" / "glic" / "Cookies"
            create_cookie_database(primary, [".youtube.com"])
            create_cookie_database(nested, ["glic.google.com"])

            self.assertEqual(primary_cookie_database(profile), primary)

    def test_session_snapshots_primary_database_and_removes_it_after_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "Default"
            create_cookie_database(
                profile / "Cookies",
                [".youtube.com", ".google.com"],
            )

            with yt_dlp_cookie_session(
                enabled=True,
                chrome_profile=str(profile),
            ) as session:
                self.assertEqual(session.youtube_cookies, 1)
                self.assertEqual(session.total_cookies, 2)
                snapshot_profile = Path(session.browser_spec.removeprefix("chrome:"))
                snapshot_database = snapshot_profile / "Cookies"
                self.assertTrue(snapshot_database.is_file())

            self.assertFalse(snapshot_profile.exists())


if __name__ == "__main__":
    unittest.main()
