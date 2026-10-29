#/bin/bash
#
# Script to run LineSeeker
# https://github.com/jigonzal/LineSeeker
#
# The README.md from Jorge explains it fairly well.
#
# Author: Leindert Boogaard

# location of the scripts
PATH_TO_LINESEEKER="./LineSeeker/"
SearchLine="$PATH_TO_LINESEEKER/SearchLine.py"
GetLineCandidates="$PATH_TO_LINESEEKER/GetLineCandidates.py"

# parameters
CUBE="/path/to/cube.fits"
MinSN=3.0
MaxSigmas=15
SURVEY_NAME="my_survey"


# STEP 1:
python $SearchLine -Cube $CUBE -MinSN $MinSN -MaxSigmas $MaxSigmas

# STEP 2 (no simulations):
python $GetLineCandidates -Cube $CUBE -MinSN $MinSN -MaxSigmas $MaxSigmas -SurveyName $SURVEY_NAME -NoStokes
