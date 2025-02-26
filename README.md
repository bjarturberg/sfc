# Discrete Event Simulation of Shop Floor
This module implements a discrete event simulation framework for basic shop‐floor control in manufacturing systems.
It is based on the sub‐chapter "Basic Shop-Floor Control Concepts" from the chapter "Production Activity Control"
  in "Manufacturing Planning and Control for Supply Chain Management" by F. Robert Jacobs.

## Example:
See exmples in _examples_ folder.
### Shop floor data
```python
from sfc import Operation, Part, Item, WorkCenter

# A shovel is manufactured on a shop floor with two work stations:
assembly_work_center = WorkCenter(ID=1, priority_rule='FIFO')
other_work_center = WorkCenter(ID=2, priority_rule='FIFO')

# The shovel consits of two parts, shaft and scoop, that need to be assembled.
# Scoop:
scoop = Part(route=[
  Operation(name='Sharpen scoop', work_center_id=other_work_center.ID, run_time=15.0),
  Operation(name='Assemble scoop', work_center_id=assembly_work_center.ID, run_time=30.0)
])
# Shaft:
shaft_part = Part(route=[
  Operation(name='Paint shaft', work_center_id=other_work_center.ID, run_time=15.0)
])
# Shovel assembly:
shovel_assembly = Part(
    route=[
        Operation(name='Shovel assambely', work_center_id=assembly_work_center.ID, run_time=25.0)
    ]
)
# Final item:
shovel = Item(
  ID=10,
  project_diagram=[scoop, shaft, shovel_assembly],
  due_date=300.0,
  order_release=100.0
)
```

### Simulation
```python
def genenere_orders(work_centers: List[WorkCenter]) -> List[Item]: ...

work_centers: List[WorkCenter] = [...]
items = generate_orders(work_centers)
simulator = DiscreteEventSimulator(items=items, work_centers=work_centers)
simulator.run(until=100000)

for item in items:
  for part in item.project_diagram:
    for operation in part.route:
      print(operation.name, operation.arrived_at, operation.started_at, operation.completed_at)
```


## TODO
- [ ] Implement priority rules.
- [ ] Use tree structure in Item.project_diagram.
- [ ] Add batch size.