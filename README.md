# DC OPF model

The DC optimal power flow (OPF) model, developed as part of the Flood Resiliency Study by the NC Policy Collaboratory, simulates the intricacies of the North Carolina grid, considering hourly flooding impacts on power system components. With 662 nodes and 790 high-voltage transmission lines, the model integrates [Gridkit's OpenStreetMap data](https://zenodo.org/records/47317) as the base for the network topology. Incorporating hourly time series data for solar, nuclear, hydropower, and electricity load, it employs Python, Pyomo, and Gurobi for a 24-hour operational horizon. The optimization process navigates generator, transmission, and nodal constraints to minimize costs while meeting demand and maintaining reserves. Decision variables encompass on/off status, electricity production, and load shedding, while special attention is given to substation equipment elevation as a hardening method during extreme weather events. The model yields hourly outputs, including generation schedules, nodal shadow costs, loss of load, and power flows, offering vital insights for proactive planning in resilient power systems and vulnerable communities.

DC OPF model allows users to explore:
* Impacts of inland flooding at distinct depths. 
* Number of nodes to retain in the system
* Mathematical formulation, users can choose to formulate the problem as follows:
  * Linear programming (LP)
  * Mixed integer linear programming (MILP)



