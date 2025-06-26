###Folder Structure
mcf_eon_simulator/
├── config/
│   ├── __init__.py
│   ├── system_config.py
│   └── physical_parameters.py
├── topology/
│   ├── __init__.py
│   ├── network_loader.py
│   └── nsfnet_mcf.txt
├── physical_layer/
│   ├── __init__.py
│   ├── gsnr_calculator.py
│   ├── icxt_calculator.py
│   ├── nli_calculator.py
│   ├── ase_calculator.py
│   └── fiber_parameters.py
├── algorithms/
│   ├── __init__.py
│   ├── csb_algorithm.py
│   ├── path_computation.py
│   └── modulation_assignment.py
├── network/
│   ├── __init__.py
│   ├── mcf_network.py
│   ├── mcf_link.py
│   ├── node.py
│   └── request.py
├── simulation/
│   ├── __init__.py
│   ├── mcf_simulator.py
│   ├── statistics.py
│   └── event_handler.py
├── utils/
│   ├── __init__.py
│   ├── helper_functions.py
│   └── lookup_tables.py
├── main.py
└── run_mcf_simulation.py
