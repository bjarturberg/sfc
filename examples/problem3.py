from sfc import Operation, Part, Item, WorkCenter, DiscreteEventSimulator, tabulate


def a():
    work_center = WorkCenter(ID=1, priority_rule="first_in_first_out")
    other_work_center = WorkCenter(ID=2, priority_rule="first_in_first_out")
    data = [
        ("A", 10, 20, 4, 25),
        ("B", 12, 18, 2, 15),
        ("C", 7, 12, 2, 16),
        ("D", 5, 12, 3, 17),
        ("E", 8, 10, 2, 12),
    ]
    items = []
    for i, (name, time, total_time, n_ops, due_date) in enumerate(data):
        operation = Operation(
            name=name,
            run_time_per_unit=float(time),
            batch_size=1,
            work_center_id=work_center.ID,
        )
        avg_time_left = (total_time - time) / (n_ops - 1)
        other_operatins = [
            Operation(
                name=f"{name}-{i}",
                run_time_per_unit=float(avg_time_left),
                batch_size=1,
                work_center_id=other_work_center.ID,
            )
            for i in range(n_ops - 1)
        ]
        item = Item(
            ID=i,
            project_diagram=[Part(route=[operation, *other_operatins])],
            due_date=due_date,
            order_release=i * 0.001,
        )
        items.append(item)

    simulation = DiscreteEventSimulator(
        work_centers=[work_center, other_work_center], items=items
    )
    simulation.run(until=1000.0)
    operations = [i.project_diagram[0].route[0] for i in items]
    sorted_operations = sorted(operations, key=lambda op: op.started_at)

    results = [
        [op.name, op.started_at, op.run_time, op.completed_at]
        for op in sorted_operations
    ]
    print("Results with FIFO")
    tabulate(results, ["job", "started", "run_time", "competed"])


def b():
    work_center = WorkCenter(ID=1, priority_rule="earliest_due_date")
    other_work_center = WorkCenter(ID=2, priority_rule="first_in_first_out")
    data = [
        ("A", 10, 20, 4, 25),
        ("B", 12, 18, 2, 15),
        ("C", 7, 12, 2, 16),
        ("D", 5, 12, 3, 17),
        ("E", 8, 10, 2, 12),
    ]
    items = []
    for i, (name, time, total_time, n_ops, due_date) in enumerate(data):
        operation = Operation(
            name=name,
            run_time_per_unit=float(time),
            batch_size=1,
            work_center_id=work_center.ID,
        )
        avg_time_left = (total_time - time) / (n_ops - 1)
        other_operations = [
            Operation(
                name=f"{name}-{i}",
                run_time_per_unit=float(avg_time_left),
                batch_size=1,
                work_center_id=other_work_center.ID,
            )
            for i in range(n_ops - 1)
        ]
        item = Item(
            ID=i,
            project_diagram=[Part(route=[operation, *other_operations])],
            due_date=due_date,
        )
        items.append(item)

    simulation = DiscreteEventSimulator(
        work_centers=[work_center, other_work_center], items=items
    )
    simulation.run(until=1000.0)
    operations = [(i.project_diagram[0].route[0], i.due_date) for i in items]
    sorted_operations = sorted(operations, key=lambda x: x[0].started_at)

    print("Results with shortest-due-date")
    results = [
        [op.name, op.started_at, op.completed_at, due_date]
        for op, due_date in sorted_operations
    ]
    tabulate(results, ["job", "started", "completed", "due_date"])


if __name__ == "__main__":
    a()
    b()
