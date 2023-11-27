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

# Flooding impacts on grid assets

The study employs spatial-temporal analysis, using variograms and kriging, to assess flooding depth across river basins. Substations become inoperable if flooding depth equals or exceeds sensitive equipment height, causing electricity outages, generator failures, and disrupted transmission lines. Flooded solar farms cease energy output. Due to limited equipment height data, scenarios from 0 to 10 feet are tested. Hourly estimates of impacted substations and solar farms guide adjustments to the DC OPF model for real-time grid response to flooding. This approach provides concise insights into spatial and temporal flooding impacts on critical grid assets.

For reference, the files used for the flooding analysis are the following:
 | File Name | Folder | Description |
|--------|-----|-----|
| Krig_Cape.py   | Spatial Kriging | A Python script that performs parallelized hourly Ordinary Kriging interpolation for flooding depth data in the Cape Fear watershed using MPI, visualizing the results and exporting them to GeoTIFF raster files.| 
| Krig_Lumber.py | Spatial Kriging | A Python script that performs parallelized hourly Ordinary Kriging interpolation for flooding depth data in the Lumber watershed using MPI, visualizing the results and exporting them to GeoTIFF raster files. | 
| Krig_Neuse.py | Spatial Kriging |  A Python script that performs parallelized hourly Ordinary Kriging interpolation for flooding depth data in the Neuse watershed using MPI, visualizing the results and exporting them to GeoTIFF raster files. | 
| Krig_TP.py | Spatial Kriging |  A Python script that performs parallelized hourly Ordinary Kriging interpolation for flooding depth data in the Tar Pamlico watershed using MPI, visualizing the results and exporting them to GeoTIFF raster files. |
| Krig_White.py | Spatial Kriging | A Python script that performs parallelized hourly Ordinary Kriging interpolation for flooding depth data in the White Oak watershed using MPI, visualizing the results and exporting them to GeoTIFF raster files. | 
| General_flood_all.py | Impacted Assets | A Python script that determines if a grid asset is flooded over a simulated time period. | 
| All_files_depth.py | Impacted Assets | A Python script that collects the hourly flooding depth of the grid assets. | 





