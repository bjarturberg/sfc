from typing import Generator, Dict, Protocol, Tuple, List

from sfc import Item, Part, Operation, WorkCenter, DiscreteEventSimulator, PriorityRules
import numpy as np


class ProductFunction(Protocol):
    """
    A protocol that defines a callable type for product functions.
    This is just used to get type analysis.

    Parameters
    ----------
    item_id : int
        The unique identifier for the item.
    order_release : float
        The release time of the order.
    due_date : float
        The due date for the item.
    batch_size : int
        The size of the batch for the item.

    Returns
    -------
    Item
        The resulting item after applying the product function.
    """
    def __call__(
        self, item_id: int, order_release: float, due_date: float, batch_size: int
    ) -> Item: ...


def order_generator(
    start_item_id: int,
    create_product_item: ProductFunction,
    avg_interarrival_time: float,
    mean_batch_size: int,
    N_orders: int,
    start_time: float = 0.0,
) -> List[Item]:
    """Generate orders.

    Parameters
    ----------
    start_item_id: order_generator assign sequential IDs beginning with start_item_id.
    create_product_item: The factory function for the specific product; see example in create_bread(...)
    avg_interarrival_time: The mean interarrival rate.
    mean_batch_size: mean batch size of prodcut orders.
    start_time: Initial release time of the first order.

    Yields
    ------
    Item instances (orders).
    """
    current_time = start_time

    item_id = start_item_id
    orders = []
    for i in range(N_orders):
        interarrival_time = generate_interarrival_time(avg_interarrival_time)
        current_time += interarrival_time

        batch_size = gen_batch_size(mean_batch_size)
        due_date = gen_due_date(
            create_product_item, batch_size=batch_size, order_release=current_time
        )

        order = create_product_item(
            item_id=item_id,
            order_release=current_time,
            due_date=due_date,
            batch_size=batch_size,
        )
        item_id += 1
        orders.append(order)
        
    return orders


def gen_due_date(
    create_product: ProductFunction,
    batch_size: int,
    order_release: float,
    slack_scale: float = 2.0,
) -> float:
    non_queue_time = create_product(
        item_id=0, order_release=order_release, due_date=0, batch_size=batch_size
    ).non_queue_time()
    random_part = 0.0  # TODO: implement random part
    due_date = order_release + slack_scale * non_queue_time + random_part
    return due_date


def gen_batch_size(mean_batch_size: int, min_batch_size: int = 1) -> int:
    random_batch_size = mean_batch_size  # TODO: implement random_batch_size, mean er ekki random
    return max(min_batch_size, random_batch_size)


def generate_interarrival_time(avg_interarrival_time: float) -> float:
    """Generate a random duration until next order."""
    return np.random.exponential(scale=avg_interarrival_time)


def generate_avg_interarrival_time(product_function, mean_batch_size):
    # TODO: Change this function to adjust the the average interrarival_time.
    # The current implementation will probably create too many orders.
    mean_product = product_function(
        item_id=0, order_release=0.0, due_date=0.0, batch_size=mean_batch_size
    )
    return mean_product.non_queue_time()


def create_bread(
    item_id: int, order_release: float, due_date: float, batch_size: int
) -> Item:
    "Bake breads."
    part_1 = Part(
        route=[
            Operation(
                ID=5,
                name="Measure ingredients",
                work_center_id=2,
                setup_time=10.0,
                run_time_per_unit=5 * batch_size,
                move_time=3.0,
                batch_size=1,
            ),
            Operation(
                ID=3,
                name="Mix dough",
                work_center_id=1,
                setup_time=30.0,
                run_time_per_unit=20 * batch_size,
                move_time=5.0,
                batch_size=1,
            ),
            Operation(
                ID=3,
                name="Shape",
                work_center_id=3,
                setup_time=10.0,
                run_time_per_unit=5,
                move_time=0.0,
                batch_size=batch_size,
            ),
            Operation(
                ID=3,
                name="Bake",
                work_center_id=4,
                setup_time=5.0,
                run_time_per_unit=40 * batch_size,
                move_time=10.0,
                batch_size=1,
            ),
        ]
    )

    part_2 = Part(
        route=[
            Operation(
                ID=5,
                name="Quality inspection",
                work_center_id=3,
                setup_time=10.0,
                run_time_per_unit=1,
                move_time=3.0,
                batch_size=batch_size,
            ),
            Operation(
                ID=5,
                name="Put bread in shelves",
                work_center_id=5,
                setup_time=50.0,
                run_time_per_unit=1,
                move_time=100.0,
                batch_size=batch_size,
            ),
        ]
    )

    return Item(
        ID=item_id,
        order_release=order_release,
        due_date=due_date,
        project_diagram=[part_1, part_2],
    )


def example_simulate_data(
    priority_rule_by_work_center_id: Dict[int, PriorityRules]
) -> Tuple[List[WorkCenter], List[Item]]:
    """Example simulation of bread baking.
    
    To simulate data for one product we need:
      1. a function (ProductionFunction) that creates one item of the product given:
        a. an ID for the item;
        b. order release time (based on interarrival times);
        c. due date;
        d. batch_size;
      2. decide what the average 
      3. a function that generates due dates based on order release and batch size;
      4. a function that generates a batch size.
      
    """
    # Pick some average batch size
    bread_avg_batch_size = 30
    # Use generate_avg_interarrival_time (and adjust it) or set the avg_interarrival_time directly.
    bread_avg_interarrival_time = generate_avg_interarrival_time(create_bread, bread_avg_batch_size)
    production_information = [
        [
            create_bread,  # 1. a function that creates one item of the product.
            bread_avg_batch_size,
            bread_avg_interarrival_time
            100 # N_orders
        ],
        # add more products here:
        # [
        #     create_bun,
        #     bun_avg_batch_size,
        #     bun_avg_interarrival_time
        # ],
        # [
        #     ...
        # ]
    ]

    start_time = 0.0
    all_orders = []
    item_id = 0
    for product_function, mean_batch_size, avg_interarrival_time, N_oders in production_information:
         product_orders = order_generator(
            item_id, product_function, avg_interarrival_time, mean_batch_size, N_orders, start_time
        )
        all_orders.extend(product_orders)
        item_id = product.orders[-1].ID + 1

        #last_order_release = 0.0
        #assert last_order_release < until
        #while last_order_release < until:
        #    item = next(product_generator)
        #    all_orders.append(item)
        #    last_order_release = item.order_release
        #item_id = item.ID + 1  # type: ignore

    
    work_centers = [
        WorkCenter(ID=1, priority_rule=priority_rule_by_work_center_id[1]),
        WorkCenter(ID=2, priority_rule=priority_rule_by_work_center_id[2]),
        WorkCenter(ID=3, priority_rule=priority_rule_by_work_center_id[3]),
        WorkCenter(ID=4, priority_rule=priority_rule_by_work_center_id[4]),
        WorkCenter(ID=5, priority_rule=priority_rule_by_work_center_id[5]),
    ]

    return work_centers, all_orders


if __name__ == "__main__":
    work_centers, items = example_simulate_data(
        priority_rule_by_work_center_id={i: "first_in_first_out" for i in range(1, 6)},
    )
    work_center_ids_in_items = set(op.work_center_id for item in items for op in item.all_operations())
    work_center_ids = set(wc.ID for wc in work_centers)
    assert work_center_ids_in_items == work_center_ids
    sim = DiscreteEventSimulator(work_centers, items)
    until = items[-1].order_release + 10000
    until = le100
    sim.run(until)
    from chart import gantt_chart
    sorted_items = sorted(items, key=lambda i.order_release)
    items_to_plot = [sorted_items[10:15]
    fig, _ = gantt_chart(items_to_plot)
    fig.savefig('bread.png')
