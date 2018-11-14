#!/bin/bash

unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
python_=python3
else python_=python
fi

installs() {
    if [[ "$unamestr" == 'Linux' ]]; then
        lsb_release -a
        sudo rm /var/lib/apt/lists/lock
        sudo rm /var/cache/apt/archives/lock
        sudo rm /var/lib/dpkg/lock
        apt-get update
        sudo rm /var/lib/apt/lists/lock
        sudo rm /var/cache/apt/archives/lock
        sudo rm /var/lib/dpkg/lock
        yes | sudo dpkg --configure -a
        yes | sudo apt update
        yes | sudo apt install python3 python3-dev
        usname=`whoami`
        python3 -m pip install pip numpy nose h5py pandas==0.23.4 scikit-learn==0.20.0 pympler Distance psutil
    fi
}

mode=$1
input_fname=$2
package_name_agg=wos_agg
package_name_gg=graph_tools
data_path=../
verb=INFO

setup_data() {
    tarfile=`ls *tar.gz`
    echo "Found tar.gz files" $tarfile "of size" $(du -smh $tarfile | awk '{print $1}')
    for tf in $tarfile; do
        tar xf $tf
    done
}

clone_repo() {
echo "starting cloning $1"
git clone https://github.com/alexander-belikov/$1.git
echo "*** list files in "$1":"
ls -lht ./$1
echo "*** list files in "$1"/"$1 ":"
ls -lht ./$1/$1
cd ./$1
echo "starting installing $1"
$python_ ./setup.py install
cd ..
}

exec_driver() {
cd ./$1
echo "starting exec_driver $1"
echo $python_
$python_ ./driver_citations.py -s $data_path -d $data_path -m $mode -v $verb -w $input_fname
cd ..
}

post_processing() {
echo "*** all files sizes :"
ls -thor *
}

# Install packages
installs
setup_data
# Clone repos from gh
clone_repo $package_name_agg
clone_repo $package_name_gg
# Execute the driver script
exec_driver $package_name_agg
# Prepare the results
post_processing
