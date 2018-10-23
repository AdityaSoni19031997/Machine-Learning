#!/bin/bash
# my first bash script to automate git stuffs..

echo "Started at `date`"

repo_dir=`pwd`
IFS=$'\n'

for repo in `ls "$repo_dir/"`
do
  if [ -d "$repo_dir/$repo" ]
  then
    echo "Updating $repo_dir/$repo at `date`"
    if [ -d "$repo_dir/$repo/.git" ]
    then
      cd "$repo_dir/$repo"
      git status
      echo "Fetching"
      git fetch
      echo "Pulling"
      git pull
    else
      echo "Skipped as it doesn't look like it's having a .git folder."
    fi
    echo "Completed at `date`"
    echo
  fi
done
