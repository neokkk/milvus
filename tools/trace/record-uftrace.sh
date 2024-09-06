#!/bin/bash

MILVUS_PATH=$HOME/milvus
LD_LIBRARY_PATH=$MILVUS_PATH/internal/core/output/lib:lib:$LD_LIBRARY_PATH uftrace record $MILVUS_PATH/bin/milvus run standalone
