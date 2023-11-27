# DC OPF model

The DC optimal power flow (OPF) model, developed as part of the Flood Resiliency Study by the NC Policy Collaboratory, simulates the intricacies of the North Carolina grid, considering hourly flooding impacts on power system components. With 662 nodes and 790 high-voltage transmission lines, the model integrates [Gridkit's OpenStreetMap data](https://zenodo.org/records/47317) as the base for the network topology. Incorporating hourly time series data for solar, nuclear, hydropower, and electricity load, it employs Python, Pyomo, and Gurobi for a 24-hour operational horizon. The optimization process navigates generator, transmission, and nodal constraints to minimize costs while meeting demand and maintaining reserves. Decision variables encompass on/off status, electricity production, and load shedding, while special attention is given to substation equipment elevation as a hardening method during extreme weather events. The model yields hourly outputs, including generation schedules, nodal shadow costs, loss of load, and power flows, offering vital insights for proactive planning in resilient power systems and vulnerable communities.

DC OPF model allows users to explore:
* Impacts of inland flooding at distinct depths. 
* Number of nodes to retain in the system
* Mathematical formulation, users can choose to formulate the problem as follows:
  * Linear programming (LP)
  * Mixed integer linear programming (MILP)
    
# Running the DC OPF model
1. **Open the File:**
   - Open the file named `MTSDataSetup.py`.

2. **Update Input Directory:**
   - Locate the line in the file specifying the directory for main inputs. Modify it to match your current working directory:
     ```python
     folder = '/home/../Inputs/'
     ```
     
3. **Replace Input Values:**
   - Substitute the values for outage data, solar power, and hourly load in the file, considering the influence of flooding on these grid assets.

4. **Save and Execute:**
   - Save the changes in "MTSDataSetup.py" and run the file.

5. **Check Output:**
   - Look for the generated output file named "MTS_data.dat."

6. **Open "wrapper.py":**
   - Open the file named `wrapper.py`.

7. **Set Output Directory:**
   - Update the location of the model outputs by replacing the `folder_2` variable.

8. **Choose Simulation Duration:**
   - Select the number of days for simulation by modifying the "days" variable.
    
9. **Select Solver:**
   - Choose a solver for the model (e.g. Gurobi, CPLEX).

10. **Run and Verify:**
    - Execute the "wrapper.py" file and check the generated outputs.

Follow these steps to configure and run the DC OPF model with the specified inputs and simulation parameters.

For reference, the files in the DC OPF model are the following:
 | File Name | Description |
|--------|-----|
| gen_mat.py   | A binary file indicating the buses to which generators are linked.| 
| line_to_bus.py  | A binary file revealing the connections between buses and their associated lines. | 
| MTS_LP.py    | This encompasses the LP problem formulation of DC OPF model.  | 
| MTS_MILP.py    | This encompasses the MILP problem formulation of DC OPF model.  |
| MTSDataSetup.py    | A Python script is created to generate an "MTS_data.dat" file, incorporating the provided information in a format compatible with Pyomo.  | 
| wrapper.py    | 	This script invokes an optimization solver, initiates the simulations, and retrieves the outputs of the model. | 




