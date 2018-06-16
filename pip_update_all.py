#simple script to update all packages in one go
#Early Warning - Will take quite a lot of time , may break your Enviornmnet..

import pip
from subprocess import call

packages = [dist.project_name for dist in pip.get_installed_distributions()]
call("pip3 install --upgrade " + ' '.join(packages), shell=True)
