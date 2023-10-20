#!/bin/bash

# Upload files in a folder

FILEPATHS=$(find ./hubert_features/train/ -name "*.pt")
DROPBOX_PATH="/DLCV/Final/b08901169/hubert_features/train/"
ACCESS_TOKEN="sl.BV2hFTf5dq4ETivCf0lSk7RqjvfV9lXiNVi5r0tFp-a5sUDLuwZmEn_d2f9rX7Br31yQxi9dCTtSYc6Mhf4LbffIcNtF55J35BHuw0uPFDTKw_6Xf9vS-SDijXaeE-SfpsYSRX8"

for FILEPATH in $FILEPATHS; do
    FILE_NAME=$(basename ${FILEPATH})

    curl -X POST https://content.dropboxapi.com/2/files/upload  \
        --header "Authorization: Bearer ${ACCESS_TOKEN}" \
        --header "Dropbox-API-Arg: {\"path\": \"${DROPBOX_PATH}${FILE_NAME}\"}" \
        --header "Content-Type: application/octet-stream" \
        --data-binary @$FILEPATH \
        --progress-bar
done

echo "upload complete"