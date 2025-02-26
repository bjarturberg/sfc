from sfc import Operation, Part, Item, WorkCenter, DiscreteEventSimulator


# A shovel is manufactured on a shop floor with two work stations:
assembly_work_center = WorkCenter(ID=1, priority_rule='FIFO')
other_work_center = WorkCenter(ID=2, priority_rule='FIFO')

# The shovel consits of three parts: handle, shaft and scoop.
# Handle
paint_handle_bracket = Operation(name='Paint handle bracket', work_center_id=other_work_center.ID, run_time=10.0)
assemble_handle = Operation(name='Assemble handle', work_center_id=assembly_work_center.ID, run_time=20.0)
handle_part = Part(route=[paint_handle_bracket, assemble_handle])
# Scoop
sharpen_blade = Operation(name='Sharpen blade', work_center_id=other_work_center.ID, run_time=15.0)
assemble_scoop = Operation(name='Assemble blade', work_center_id=assembly_work_center.ID, run_time=30.0)
blade_part = Part(route=[sharpen_blade, assemble_scoop])
# Shaft
paint_shaft = Operation(name='Paint shaft', work_center_id=other_work_center.ID, run_time=15.0)
shaft_part = Part(route=[paint_shaft])

# Shovel assembly
shovel_assembly = Part(
    route=[
        Operation(name='Shovel assambely', work_center_id=assembly_work_center.ID, run_time=25.0)
    ]
)


shovel = Item(ID=10, project_diagram=[handle_part, blade_part, shaft_part, shovel_assembly], due_date=300.0, order_release=200.0)
