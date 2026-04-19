import unittest

from finance_agent.contracts.common import AuthMode
from finance_agent.orchestration.identity import HeaderIdentityResolver, LocalIdentityResolver


class IdentityResolverTests(unittest.TestCase):
    def test_local_resolver_returns_static_principal(self) -> None:
        resolver = LocalIdentityResolver(principal_id="local_poc")
        identity = resolver.resolve()
        self.assertEqual(identity.principal_id, "local_poc")
        self.assertEqual(identity.auth_mode, AuthMode.LOCAL)

    def test_header_resolver_extracts_principal(self) -> None:
        resolver = HeaderIdentityResolver()
        identity = resolver.resolve({"X-Principal-Id": "user-42"})
        self.assertEqual(identity.principal_id, "user-42")
        self.assertEqual(identity.auth_mode, AuthMode.API)

    def test_header_resolver_requires_header(self) -> None:
        resolver = HeaderIdentityResolver()
        with self.assertRaisesRegex(ValueError, "Missing principal id"):
            resolver.resolve({})


if __name__ == "__main__":
    unittest.main()
