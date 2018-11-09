#!/bin/bash

unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
python_=python3
else python_=python
fi

py_code="import sys
with open(sys.argv[1]) as f:
    content = f.readlines()
    content = [x.strip() for x in content]
inds = []
for x,j in zip(content, range(len(content))):
    if x == '[global]':
        inds.append(j)
content = content[:inds[1]]
content.insert(inds[0]+1, sys.argv[2])
with open(sys.argv[1], 'w') as f:
    f.write('\n'.join(content) + '\n')"

installs() {
    if [[ "$unamestr" == 'Linux' ]]; then
        lsb_release -a
        apt-get update
        sudo rm /var/lib/apt/lists/lock
        sudo rm /var/cache/apt/archives/lock
        sudo rm /var/lib/dpkg/lock
        yes | sudo dpkg --configure -a
        yes | sudo apt-get install python3-h5py python3-nose python3-seaborn python3-nose-parameterized
        usname=`whoami`
        fstring="base_compiledir=/tmp/$usname/theano.NOBACKUP"
        python -c "$py_code" ~/.theanorc $fstring
    fi
}

git_ab=alexander-belikov
git_kl=KnowledgeLab

package_name_dh=datahelpers
package_name_bm=BigMech
package_name_mc=pymc3

orig=$1
ver=$2
nsamples=$3
min_seq=$4
nproc=$5
draws=$6
begin=$7
end=$8
low=$9
hi=${10}
dry=${11}

spec="year identity ai pos"
spec2="year_identity_ai_pos"
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
echo "pwd"
pwd
ls -lht ../../*
$python_ runner_timestep_mp.py -s $nsamples -d $spec -n $draws -p $nproc \
    --datapath ../../ --reportspath ../../ --figspath ../../ \
    --tracespath ../../ --logspath ../../ \
    --func model_f --partition-sequence $low $hi --minsize-sequence $min_seq \
    --version $ver --origin $orig \
    -b $begin -e $end \
    --dry $dry
cd ../..
}

post_processing() {
suffix="${orig}_v_${ver}_c_${spec2}_m_${nsamples}_n_${min_seq}_a_${low}_b_${hi}"
tar -czf figs.tar.gz fig_$suffix*.pdf
tar -czf traces.tar.gz trace_$suffix*.pgz
tar -czf logs.tar.gz runner_*.log
tar -czf reports.tar.gz report_$suffix*.pgz
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
cd ./$package_name_mc
git checkout 6b3994113cae965c8717f90007e72052a39743cf
cd ..
clone_repo $package_name_bm $git_kl
clone_repo $package_name_dh $git_ab
# Execute the driver script
exec_driver $package_name_bm
# Prepare the results
post_processing
