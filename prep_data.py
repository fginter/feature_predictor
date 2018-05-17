from data import read_conll, build_dicts
import sys
import argparse
import json

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='prep_dicts')
    args = parser.parse_args()

    dicts=build_dicts(sys.stdin)
    json.dump(dicts,sys.stdout,indent=4)
    
