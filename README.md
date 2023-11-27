# DC OPF model

The DC optimal power flow (OPF) model, developed as part of the Flood Resiliency Study by the NC Policy Collaboratory, simulates the intricacies of the North Carolina grid, considering hourly flooding impacts on power system components. With 662 nodes and 790 high-voltage transmission lines, the model integrates [Gridkit's OpenStreetMap data](https://zenodo.org/records/47317) as the base for the network topology. Incorporating hourly time series data for solar, nuclear, hydropower, and electricity load, it employs Python, Pyomo, and Gurobi for a 24-hour operational horizon. The optimization process navigates generator, transmission, and nodal constraints to minimize costs while meeting demand and maintaining reserves. Decision variables encompass on/off status, electricity production, and load shedding, while special attention is given to substation equipment elevation as a hardening method during extreme weather events. The model yields hourly outputs, including generation schedules, nodal shadow costs, loss of load, and power flows, offering vital insights for proactive planning in resilient power systems and vulnerable communities.

DC OPF model allows users to explore:
* Impacts of inland flooding at distinct depths. 
* Number of nodes to retain in the system
* Mathematical formulation, users can choose to formulate the problem as follows:
  * Linear programming (LP)
  * Mixed integer linear programming (MILP)

For reference, in the DC OPF model are the following:
 | File Name | Description |
|--------|-----|
| gen_mat.py   | Binary file showing which generators are connected to which buses | 
| line_to_bus.py  | Binary file showing which lines are connected to which buses | 
| MTS_LP.py    | This contains the LP problem formulation of DC OPF model.  | 
| MTS_MILP.py    | This contains the MILP problem formulation of DC OPF model.  |
| MTSDataSetup.py    | Python script that creates "MTS_data.dat" file which includes all data above in a format accessible by Pyomo  | 
| wrapper.py    | 	This script calls an optimization solver, starts the simulations, and returns the model outputs.  | 




