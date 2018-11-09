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
        pip3 install Distance
        usname=`whoami`
    fi
}

git_ab=alexander-belikov
git_kl=KnowledgeLab

package_name_dh=datahelpers
package_name_bm=BigMech
package_name_mc=pymc3

input_rank=$1
input_pmids=$2
output_file=$3
np=$4
head=$5

under=_

setup_data() {
    tarfile=`ls *tar.gz`
    echo "Found tar.gz files" $tarfile "of size" $(du -smh $tarfile | awk '{print $1}')
    for tf in $tarfile; do
        tar xf $tf
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

$python_ generate_affiliation_ranking.py -d $head -p $np --fname-output $output_file \
    --fname-rankings $input_rank --fname-articles $input_pmids
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
# setup_data
# Clone repos from gh
clone_repo $package_name_mc $git_ab
clone_repo $package_name_bm $git_kl
clone_repo $package_name_dh $git_ab
exec_driver $package_name_dh
# Prepare the results
post_processing
