# RTK v3 migration guide

## Use `--size` instead of `--dimension`

As of RTK 3.0, the `--dimension` argument in applications is deprecated and
will be removed in a future release.  Please use the `--size` argument instead.
This update aligns RTK with upstream ITK's member name, where `--size` more
accurately reflects the expected parameter (e.g., image size) and avoids
confusion with spatial dimensions (e.g., 2D vs 3D).

## Forbild phantom file format only

The legacy `rtk::GeometricPhantomFileReader` has been removed starting with RTK
3.0. This handcrafted file format was not documented and everything it could do
is also achievable with the only file format for geometric phantoms accepted in
RTK 3.0, the [Forbild phantom file format](Phantom.md). The flag `--forbild`
has also been removed from command line tools as it is now the default and only
option.
