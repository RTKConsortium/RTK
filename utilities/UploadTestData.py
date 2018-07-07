#!/usr/bin/env python

import argparse
import girder_client
from girder_client import GirderClient
import os
import fnmatch
from distutils.version import StrictVersion

if StrictVersion(girder_client.__version__) < StrictVersion("2.0.0"):
    raise Exception("Girder 2.0.0 or newer is required")

class GirderExternalDataCli(GirderClient):
    """
    A command line Python client for interacting with a Girder instance's
    RESTful api, specifically for performing uploads into a Girder instance.
    """
    def __init__(self, apiKey):
        """initialization function to create a GirderCli instance, will attempt
        to authenticate with the designated Girder instance.
        """
        GirderClient.__init__(self,
                              apiUrl='https://data.kitware.com/api/v1')

        self.authenticate(apiKey=apiKey)

    def _uploadContentLinkItem(self, content_link, test_folder,
            parent_id, reuseExisting=True, dryRun=False,):
        """Upload objects corresponding to CMake ExternalData content links.

        This will upload the file for the content link *content_link*
        located in the *test_folder* folder, to the corresponding Girder folder.

        :param parentType: one of (collection,folder,user), default of folder.
        :param reuseExisting: bool whether to accept an existing item of
            the same name in the same location, or create a new one instead.
        :param dryRun: Do not actually upload any content.
        """
        test_folder = os.path.normpath(test_folder)
        name = os.path.splitext(content_link)[0] #remove .sha512 extension

        if dryRun:
            # create a dryRun placeholder
            folder = {'_id': 'dryRun'}
            print("\n Trying to upload \"" + name + " (" + content_link + ")\" into folder \"" + self.getFolder(parent_id)['name'] + test_folder + "\"")
        else:
            parent_folder = parent_id
            for dirnames in test_folder.split("/"):
                if dirnames:
                    folder = self.loadOrCreateFolder(dirnames, parent_folder, 'folder')
                    parent_folder = folder['_id']

        scriptDir = os.path.dirname(os.path.realpath(__file__))
        workDir = os.path.normpath(scriptDir + "/../test/" + test_folder)

        content_link = os.path.join(workDir, content_link)
        
        if os.path.isfile(content_link) and \
                fnmatch.fnmatch(content_link, '*.sha512'):
            with open(content_link, 'r') as fp:
                hash_value = fp.readline().strip()

            content_file = os.path.join(workDir, ".ExternalData_SHA512_" + hash_value)
            if os.path.isfile(content_file):
                print("\n Found content file: " + content_file)
                    
                self._uploadAsItem(
                    name,
                    folder['_id'],
                    content_file,
                    reuseExisting=reuseExisting,
                    dryRun=dryRun)

                print("\n Upload Successful !")
                return

        print(" File not found. Upload failed for " + content_link)


def main():
    parser = argparse.ArgumentParser(
        description='Upload CMake ExternalData content links to Girder')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='will not write anything to Girder, only report on what would '
        'happen')
    parser.add_argument('--api-key', required=True, default=None)
    # Default is RTK/RTKTestingData
    parser.add_argument('--parent-id', required=False,
                        default='5b178a918d777f15ebe1fff8',
                        help='id of Girder parent target (default: RTK/RTKTestingData)')
    parser.add_argument('--content-link', required=True,
                        help='name of the content-link .sha512 file. (eg: Test.png.sha512)')
    parser.add_argument('--test-folder', required=True,
                        help='path to the content link file. (eg: Baseline/path/To/File/)')
    args = parser.parse_args()

    gc = GirderExternalDataCli(args.api_key)
    gc._uploadContentLinkItem(args.content_link, args.test_folder, args.parent_id,
            reuseExisting=True, dryRun=args.dry_run)

if __name__ == '__main__':
    main()
