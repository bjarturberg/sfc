from typing import List

import pandas as pd

bom = pd.read_csv("h3_bom.csv")
orders = pd.read_csv("h3_orders.csv")


# We want to generate items using bom and orders.
# We want to create one item per order (row) in orders.
# Each item is generated using information from n rows in bom
# where n is equal to them number of operations needed to create
# the item.
# orders.itemID maps to bom.product_id

# Lets start by creating a function that accepts an order
# and generates an item based on product_id


def create_item(
    product_id: int, due_data: float, order_relase: float, batch_size: int
) -> Item:
    # Use bom to create the item by selecting the right rows using product_id.
    pass


def create_all_orders() -> List[Item]:
    # Loop through rows in orders.csv and create one item for each row.
    pass
