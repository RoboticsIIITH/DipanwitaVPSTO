# Planning in PyBullet using Via-Point Stochastic Optimization(VP-STO)

Code for Via-Point Stochastic Optimization(VP-STO) planner for UR5e manipulator. Simulated in PyBullet using `pybullet_planning` library.

## Instructions to Run

* install `pybullet_planning`. Run the following:

```
pip install git+https://github.com/yijiangh/pybullet_planning@dev#egg=pybullet_planning
```

This will install both [pybullet](https://github.com/bulletphysics/bullet3) and [pybullet_planning](https://github.com/yijiangh/pybullet_planning).

We also use `termcolor` to do colorful terminal printing, install it by:
```
pip install termcolor
```

* install VP-STO dependencies from [https://github.com/JuJankowski/vp-sto](https://github.com/JuJankowski/vp-sto)

* install VP-STO. Run the following:

```
cd vp-sto
pip install .
```

* `cd examples`
* Run the vpsto code:

```
python ur5_vpsto.py
```