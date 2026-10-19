from langchain_core.tools import tool

# Mock database. In a real system this would be an API or a SQL query.
ORDERS = {
    "1001": {"user_id": "u1", "status": "shipped", "carrier": "DHL",
             "eta": "2026-10-22", "items": ["Wireless Mouse"], "total": 29.99},
    "1002": {"user_id": "u1", "status": "processing", "carrier": None,
             "eta": "2026-10-25", "items": ["Keyboard", "USB Cable"], "total": 64.50},
    "2001": {"user_id": "u2", "status": "delivered", "carrier": "UPS",
             "eta": "2026-10-15", "items": ["Monitor"], "total": 199.00},
}

NOT_FOUND = "No order with that ID was found for this account."


def make_order_tools(user_id: str) -> list:
    """Create tools bound to ONE logged-in user.

    user_id is fixed here by our code (from the login session). The LLM can never
    see or change it, so it cannot read another customer's orders.
    """

    @tool
    def get_order_status(order_id: str) -> str:
        """Look up the status of the current customer's order.
        order_id is the order number, digits only (for example 1001)."""
        oid = order_id.strip().lstrip("#")
        order = ORDERS.get(oid)

        # Same message for "doesn't exist" and "belongs to someone else",
        # so an attacker cannot discover which order IDs exist.
        if order is None or order["user_id"] != user_id:
            return NOT_FOUND

        parts = [
            f"Order {oid}: status={order['status']}",
            f"items={', '.join(order['items'])}",
            f"total=${order['total']:.2f}",
            f"estimated date={order['eta']}",
        ]
        if order["carrier"]:
            parts.append(f"carrier={order['carrier']}")
        return "; ".join(parts)

    return [get_order_status]


if __name__ == "__main__":
    # Test the tool as plain Python first, with NO LLM involved.
    tools = make_order_tools("u1")
    get_order_status = tools[0]

    print(get_order_status.invoke({"order_id": "1001"}))     # u1's own order
    print(get_order_status.invoke({"order_id": "#1002"}))    # '#' is handled
    print(get_order_status.invoke({"order_id": "2001"}))     # belongs to u2 -> blocked
    print(get_order_status.invoke({"order_id": "9999"}))     # does not exist