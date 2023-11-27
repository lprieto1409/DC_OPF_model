# DC OPF model

The direct current (DC) optimal power flow (OPF) model is created through the Flood Resiliency Study by NC Policy Collaboratory. This model is designed to simulate the North Carolina grid behavior while accounting for the hourly impacts of flooding on power system components across space. The model represents a network of 662 nodes and 790 high-voltage transmission lines, and it uses Gridkit's OpenStreetMap data and geographic information from various public sources. Additionally, it utilizes hourly time series data for solar, nuclear, hydropower, and electricity load to minimize costs while meeting hourly electricity demand and maintaining operational reserves. The model is Python-based, employing Pyomo and Gurobi, and it operates on a 24-hour horizon, providing hourly outputs such as generation schedules, nodal shadow costs, loss of load, and power flows.

DC OPF model allows user to explore:
* Impacts of inland flooding at distinct depths. 
* Number of nodes to retain in the system
* Mathematical formulation, user can choose to formulate the problem as:
  * Linear programming (LP)
  * Mixed integer linear programming (MILP)



