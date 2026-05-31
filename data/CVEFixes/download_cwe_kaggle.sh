#!/bin/bash
curl -L -o cvefixes-vulnerable-and-fixed-code.zip\
  https://www.kaggle.com/api/v1/datasets/download/girish17019/cvefixes-vulnerable-and-fixed-code

unzip cvefixes-vulnerable-and-fixed-code.zip -d cvefixes-vulnerable-and-fixed-code
rm cvefixes-vulnerable-and-fixed-code.zip
mv cvefixes-vulnerable-and-fixed-code/*.csv .
rm -rf cvefixes-vulnerable-and-fixed-code