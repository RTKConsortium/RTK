
# How to prepare an RTK release

Check ITK's instructions and update this wiki page accordingly.

## Prepare release message

Based on previous messages (`git show tags`).

```
git log --no-merges v2.4.0..HEAD --pretty=format:"%h; %an; %ad: %s" >release_notes.txt
git log --no-merges v2.4.0..HEAD --pretty=format:"%an" | sort -u >contributors.txt
```

## Prepare repository

* Modify the RTK release number(s) in `CMakeLists.txt`,
* Modify the RTK release number and the required ITK version for Python packages in `setup.py`.
* Tag git repository on this change with a copy of the release message.

## Backup Doxygen documentation

```
ctest -S /home/srit/src/rtk/rtk-dashboard/russula_doxygen.cmake -V
cd /home/srit/src/rtk/RTK-website/
mv  /home/srit/src/rtk/dashboard_tests/RTK-Doxygen/Doxygen/html Doxygen241
./sync.sh
```

## Prepare website

Create news message in `RTK/news`, update `rtkindex.html` and `RTK/news/lefthand.html` accordingly.

Update page `RTK/resources/software.html`.

## Advertise

Post message on the mailing list [`rtk-users@openrtk.org`](mailto:rtk-users@openrtk.org).

## Verify binaries

Binaries will now be posted by GitHub actions when tagging the repository, simply verify that the Github action goes well and that the packages have been posted once it is over. 
