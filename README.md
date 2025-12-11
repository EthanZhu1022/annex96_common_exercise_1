# IEA EBC Annex 96: Common Exercise 1

## Instructions (December 12, 2025)

---

## Table of Contents
- [Objective](#objective)
- [Instructions](#instructions)
- [CityLearn Tutorial](#citylearn-tutorial)
- [Suggested Timeline](#suggested-timeline)
- [Evaluation Metrics](#evaluation-metrics)
  - [Reference Load Profile](#reference-load-profile)
  - [Primary Metrics](#primary-metrics)
  - [Secondary Metrics](#secondary-metrics)

---

## Objective

The purpose of **Common Exercise 1** is to bring together all participating Annex 96 research groups in a joint challenge using a **shared toolchain (CityLearn)**.  
Although flexible demand can support many grid services, the first exercise focuses on a single scenario:

> **A portfolio of buildings must track a daily portfolio-level reference load profile purchased by an aggregator.**

This allows for consistent comparison across different control strategies and architectures.

---

## Instructions

1. **Scope**  
   The exercise focuses on grid-interactive efficient buildings with:  
   - electrically driven **space heating** or **space cooling**,
   - **on-site solar PV**,  
   - **battery electric storage (BES)**.

2. **Select a Climate / Building Portfolio**  
   Choose **one** of the two provided building portfolios (both from the ResStock dataset) found in data/datasets/annex96_ce1_{state}_neighborhood:  
   - **Texas** ('TX': cooling-dominated)  
   - **Vermont** ('VT': heating-dominated)  
   You may experiment with both.

3. **Portfolio Description**  
   Each portfolio contains **25 single-family homes**, each pre-configured with PV + BES in CityLearn.  
   All participants use the same buildings for consistency.

4. **Degrees of Freedom (Control Actions)**  
   Controllers can manipulate:
   - **Battery charge/discharge**,  
   - **Heat pump operation** (thermal mass control).  
   PV **cannot** be exported to the grid.

5. **Primary Task**  
   Develop any control architecture—centralized, decentralized, hierarchical, distributed, RL, MPC, RBC, or hybrids—to track a **daily portfolio-level reference load profile**.

6. **Secondary Tasks**  
   Monitor and report building-level and portfolio-level metrics:  
   - operational cost  
   - emissions  
   - ramping  
   - other relevant metrics (signals provided)

7. **Training Period**  
   One month:  
   - **January** (Vermont, peak heating)  
   - **August** (Texas, peak cooling)  
   You may re-simulate the month repeatedly until convergence.

8. **Testing Period**  
   One month of unseen data following training:  
   - **February** for Vermont  
   - **September** for Texas  
   All performance metrics are evaluated on test results.

9. **Baseline Controller**  
   Baseline scenario assumes:
   - PV system **without** battery storage,  
   - thermostat **deadband** controller,  
   - PV self-consumption only (no export).  

---
## CityLearn Tutorial
See /notebooks/ce1_tutorial for a quickstart on how to use citylearn with a standard RBC.

---
## Suggested Timeline

- **Finalize Common Exercise + datasets**: December 12, 2025
- **Kickoff**: January 16, 2025
- **Submission Deadline**: April 15, 2025
- **Results comparison**: next Annex 96 meeting in Torino

---

## Evaluation Metrics

To ensure comparability across teams, the following metrics should be reported at a minimum.  
Simulations must include **all end uses** (HVAC, appliances, lighting, plug loads) to compute both **building-level** and **portfolio-level** metrics.

Metrics should be evaluated **daily** over the testing month, producing distributions or box plots where appropriate.


### Reference Load Profile

The main objective is to track the **aggregated load profile** of the building portfolio.

The reference profile for each day is defined as:

> **The daily average aggregated baseline load of the portfolio**,  
> resulting in a constant target value for that day.

**Example:**  
If daily total load = 2400 kWh:  
- reference power = 2400 kWh / 24 h = **100 kW**  
- reference 15-minute energy = 2400 kWh / 96 intervals = **25 kWh / interval**

This becomes the target profile for the entire day.



## Primary Metrics

### Portfolio-Level Reference Load Tracking

#### 1. Normalized Mean Bias Error (NMBE) [%]  
Measures systematic over- or under-consumption:

\[
	ext{NMBE} = rac{	ext{mean}(y - y_	ext{ref})}{	ext{mean}(y_	ext{ref})} 	imes 100\%
\]

#### 2. Coefficient of Variation of RMSE (CV-RMSE) [%]  
Captures total tracking error (bias + variance):

\[
	ext{CV-RMSE} = rac{\sqrt{	ext{mean}((y - y_	ext{ref})^2)}}{	ext{mean}(y_	ext{ref})} 	imes 100\%
\]


### Thermal Comfort

#### Temperature Exceedance Hours (hours)  
Hours where indoor temperature is outside comfort bounds:

- **Heating season:** 20–24 °C  
- **Cooling season:** 22–26 °C  

Report:  
- total exceedance hours per building  
- percent of hours  
- distribution across portfolio  


## Secondary Metrics

### Fairness
Assesses how evenly flexibility actions are distributed across buildings.  
A few buildings providing most flexibility is undesirable for:

- reliability  
- device longevity  
- occupant comfort  


### Cost Changes [%]
Relative savings between baseline/RBC and flexibility-enabled operation.


### Carbon Emissions [kg CO₂e]
Compute using hourly grid carbon intensity.


### Change in Site Total Energy Consumption [%]
Difference in total site energy between flexible and baseline scenarios.  
Indicates energy efficiency impacts of flexible operation.

### Peak Demand [kW]
Maximum 15-minute demand over the simulation.

Report:  
- peak value  
- timestamp  
- baseline vs flexible scenario results

### Peak-to-Valley Ratio [%]
\[
rac{	ext{daily peak}}{	ext{daily minimum}}
\]

Indicates required generation dispatch range.

### Load Factor [%]
\[
rac{	ext{average demand}}{	ext{peak demand}}
\]

A higher load factor indicates better utilization of generation assets.

### System Ramping [kW]
Sum of absolute changes in demand between consecutive 15-minute intervals over a day:

\[
\sum_{i=1}^{96} |y_i - y_{i-1}|
\]

Report baseline vs flexible scenario over ~30 days.

----------------------------------------------------------------


# CityLearn
CityLearn is an open source Farama Foundation Gymnasium environment for the implementation of Multi-Agent Reinforcement Learning (RL) for building energy coordination and demand response in cities. A major challenge for RL in demand response is the ability to compare algorithm performance. Thus, CityLearn facilitates and standardizes the evaluation of RL agents such that different algorithms can be easily compared with each other.

![Demand-response](https://github.com/intelligent-environments-lab/CityLearn/blob/master/assets/images/dr.jpg)

## Environment Overview

CityLearn includes energy models of buildings and distributed energy resources (DER) including air-to-water heat pumps, electric heaters and batteries. A collection of building energy models makes up a virtual district (a.k.a neighborhood or community). In each building, space cooling, space heating and domestic hot water end-use loads may be independently satisfied through air-to-water heat pumps. Alternatively, space heating and domestic hot water loads can be satisfied through electric heaters.

![Citylearn](https://github.com/intelligent-environments-lab/CityLearn/blob/master/assets/images/environment.jpg)

## Installation
Install latest release in PyPi with `pip`:
```console
pip install CityLearn
```

## Documentation
Refer to the [docs](https://intelligent-environments-lab.github.io/CityLearn/).

## CityLearn UI

CityLearn UI is a visual dashboard for exploring simulation data generated by the CityLearn framework. It was developed to simplify the analysis of results from smart energy communities, district energy coordination, demand response (among other applications), allowing users to visually inspect building-level components, compare simulation KPIs, and create simulation schemas with ease.

The interface is available in two options:

* Web app: https://citylearnui.netlify.app/ (free hosted version — not recommended for sensitive/personal data)
* Open-source code: https://github.com/Soft-CPS-Research-Group/citylearn-ui

You can check a tutorial at the official CityLearn [website](https://intelligent-environments-lab.github.io/CityLearn/ui.html), in the CityLearn UI repository [README](https://github.com/Soft-CPS-Research-Group/citylearn-ui), or at the help [tooltip of the oficial webapp](https://citylearn-ui.netlify.app/admin/help).

**Compatibility:** This version of the UI currently supports CityLearn v2.5.0 simulation data.

**Developed by:** José, a member of the [SoftCPS](https://www2.isep.ipp.pt/softcps/), Software for Cyber-Physical Systems research group (ISEP, Portugal) in collaboration with the Intelligent Environments Lab, University of Texas at Austin.

# End of README
