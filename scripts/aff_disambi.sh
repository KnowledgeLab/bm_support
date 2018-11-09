#!/bin/bash

unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
python_=python3
else python_=python
fi

installs() {
    if [[ "$unamestr" == 'Linux' ]]; then
        lsb_release -a
        apt-get update
        sudo rm /var/lib/apt/lists/lock
        sudo rm /var/cache/apt/archives/lock
        sudo rm /var/lib/dpkg/lock
        yes | sudo dpkg --configure -a
        yes | sudo apt-get install python3-h5py python3-nose python3-seaborn python3-nose-parameterized python3-nltk python3-unidecode
        pip3 install pathos Distance python-Levenshtein
        usname=`whoami`
    fi
}

git_ab=alexander-belikov
git_kl=KnowledgeLab

package_name_dh=datahelpers
package_name_bm=BigMech

spath=$1
dpath=$2
ntest=$3
nproc=$4

under=_

setup_data() {
    tarfile=`ls *tar.gz`
    echo "Found tar.gz files" $tarfile "of size" $(du -smh $tarfile | awk '{print $1}')
    for tf in $tarfile; do
        tar -xzvf $tf
    done
}

clone_repo() {
echo "starting cloning $1"
git clone https://github.com/$2/$1.git
echo "*** list files in "$1":"
ls -lht ./$1
cd ./$1
echo "starting installing $1"
$python_ ./setup.py install
cd ..
}

exec_driver() {
cd ./$1/runners
echo "starting exec_driver $1"
echo $python_
ls -thor *
$python_ runner_cluster_affs.py -s $spath -d $dpath -n $nproc --test $ntest
cd ../..
}

post_processing() {
echo "pwd"
pwd
echo "*** all files sizes :"
ls -thor *
}

# check what is here
ls -lht *
# Install packages
installs
# Setup the data files
setup_data
# Clone repos from gh
clone_repo $package_name_dh $git_ab
clone_repo $package_name_bm $git_kl
exec_driver $package_name_bm
# Prepare the results
post_processing
