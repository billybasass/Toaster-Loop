"""Dashboard API domain packages (split out of the former app.py monolith).

Each module in this package defines one mixin class of route handlers;
``DashboardApp`` in ``..app`` composes them and owns routing + dependency
wiring. Handlers are resolved by name through attribute lookup, so a domain
file is independently readable and testable while dispatch stays untouched.
"""
