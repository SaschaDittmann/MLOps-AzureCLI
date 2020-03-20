#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# -e: immediately exit if any command has a non-zero exit status
# -o: prevents errors in a pipeline from being masked
# IFS new value is less likely to cause confusing bugs when looping arrays or arguments (e.g. $@)

usage() { echo "Usage: $0 -s <subscriptionId> -g <resourceGroupName> -w <workspaceName>" 1>&2; exit 1; }

declare subscriptionId=""
declare resourceGroupName="mlops-demo"
declare workspaceName="mlops-mlwrksp"

# Initialize parameters specified from command line
while getopts ":s:g:w:h" arg; do
	case "${arg}" in
		s)
			subscriptionId=${OPTARG}
			;;
		g)
			resourceGroupName=${OPTARG}
			;;
		w)
			workspaceName=${OPTARG}
			;;
		h)
			usage
			;;
		?) 
			echo "Unknown option ${arg}"
			;;
		esac
done
shift $((OPTIND-1))

#Prompt for parameters is some required parameters are missing
if [[ -z "$subscriptionId" ]]; then
	echo "Your subscription ID can be looked up with the CLI using: az account show --out json "
	echo "Enter your subscription ID:"
	read subscriptionId
	[[ "${subscriptionId:?}" ]]
fi

if [[ -z "$resourceGroupName" ]]; then
	echo "This script will look for an existing resource group "
	echo "You can create new resource groups with the CLI using: az group create "
	echo "Enter a resource group name:"
	read resourceGroupName
	[[ "${resourceGroupName:?}" ]]
fi

if [[ -z "$workspaceName" ]]; then
	echo "This script will look for an existing Azure Machine Learning Workspace "
	echo "You can list all workspaces with the CLI using: az ml workspace list "
	echo "Enter a resource group name"
	read workspaceName
fi

if [ -z "$subscriptionId" ] || [ -z "$resourceGroupName" ] || [ -z "$workspaceName" ]; then
	echo "Either one of subscriptionId, resourceGroupName, workspaceName is empty"
	usage
fi

#login to azure using your credentials
az account show 1> /dev/null

if [ $? != 0 ];
then
	az login
fi

#set the default subscription id
az account set --subscription $subscriptionId

set +e

echo "Checking Azure CLI Extensions..."
if [ -z "$(az extension list -o tsv | grep azure-cli-ml)" ]; then
	echo "This scripts required the Azure CLI extention for machine learning."
    echo "It can me installed by using this CLI command: az extension add -n azure-cli-ml" 
	exit 0
fi

echo "Azure Machine Learning: Removing Deloyments..."
az ml service list -g "$resourceGroupName" -w "$workspaceName" -o tsv | awk '{print $3}' | xargs -I {} az ml service delete -g "$resourceGroupName" -w "$workspaceName" -n {}

echo "Azure Machine Learning: Removing Azure Kubernetes Services..."
az ml computetarget list -g "$resourceGroupName" -w "$workspaceName" -o tsv | grep aks | awk '{print $2}' | xargs -I {} az ml computetarget delete -g "$resourceGroupName" -w "$workspaceName" -n {}

echo "Azure Machine Learning: Removing Registered Models..."
az ml model list -g "$resourceGroupName" -w "$workspaceName" -o tsv | awk '{print $4}' | xargs -I {} az ml model delete -g "$resourceGroupName" -w "$workspaceName" -i {}

echo "Cleanup completed."
