# MLWorks  

This Project is a collection of code/report/ipynb for homework/project of ML Course.

## Virtual Environment

For the sake of reproducibility and reducing incompatibility, all library installation and development should be done in virtual environment, with the dependencies logged in requirements.txt shared via git.

### Recreation of virtual environment

Inside this project folder, the empty virtual environment can be created via

```cmd
py -3 -m venv .venv # Windows
python3.x -m venv .venv # Linux/Mac
```

Once the virtual environment is created, run

```cmd
.venv\Scripts\activate # Windows
source .venv/bin/activate # Linux/Mac
```

to activate the virtual environment. and run `deactivate` to leave the virtual environment. **All the following commands should be run inside the virtual environment.**

Install the wanted dependency via

```cmd
python -m pip install NAME
```

Other style/method of installation and package manager (e.g. conda, pipx, without '-m') is not recommended, as it may cause incompatibility issues.  

.venv folder can be shared via git, but the list of dependencies logged in requirements.txt is shared. If you add a new package, run the following command to update the requirements.txt.

```cmd
python -m pip freeze > requirements.txt
```

To install all the dependencies listed in requirements.txt, run

```cmd
python -m pip install -r requirements.txt
```

### Virtual environment in VSCode  

You can select interpreter (and thus virtual envrionmnt) in VSCode the command of 
`Python: Select Interpreter` and select the python.exe inside the .venv folder. for detailed instruction, see [here](https://code.visualstudio.com/docs/python/environments).

### Reference

For the reasons behind this decision, see [here](https://www.bitecode.dev/p/why-not-tell-people-to-simply-use)[] and [here](https://www.bitecode.dev/p/back-to-basics-with-pip-and-venv).

For more information about virtual environment, see [Python Official Doc](https://docs.python.org/3/library/venv.html).

## Folder Structure
