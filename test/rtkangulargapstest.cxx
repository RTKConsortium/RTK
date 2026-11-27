#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "rtkThreeDCircularProjectionGeometry.h"

/**
 * \file rtkangulargapstest.cxx
 *
 * \brief Tests rtk::ThreeDCircularProjectionGeometry::GetAngularGaps
 *
 * Compares the angular gaps of full and short scans with regular, irregular,
 * sparse, unordered and repeated angles to the expected angular gaps.
 *
 * \author Simon Rit
 */

namespace
{
constexpr double tolerance = 1e-9;

rtk::ThreeDCircularProjectionGeometry::Pointer
CreateGeometry(const std::vector<double> & anglesDeg, double sdd = 1536.)
{
  auto geometry = rtk::ThreeDCircularProjectionGeometry::New();
  for (const double angle : anglesDeg)
    geometry->AddProjection(1000., sdd, angle);
  return geometry;
}

bool
CheckValue(const std::string & testName, const std::string & what, double value, double expected)
{
  if (!(std::abs(value - expected) <= tolerance))
  {
    std::cerr << testName << ": " << what << " is " << value * itk::Math::deg_per_rad << " degrees instead of "
              << expected * itk::Math::deg_per_rad << " degrees." << std::endl;
    return false;
  }
  return true;
}

// Compares the angular gaps of distinct angles, in degrees and in circular
// order, to the expected angular gaps: half the distance between the
// previous and the next angles, except for the first and last angles of a
// short scan which are weighted by the distance to their neighbor in the scan.
// The projections are acquired in a shuffled order to check the indexing.
bool
CheckAngularGaps(const std::string &         testName,
                 const std::vector<double> & sortedAnglesDeg,
                 bool                        shortScan,
                 double                      thresholdDeg = 20.)
{
  const unsigned int nProj = sortedAnglesDeg.size();

  // Gap between each sorted angle and the next one, wrapping around 360
  std::vector<double> nextGaps(nProj);
  for (unsigned int i = 0; i < nProj; i++)
  {
    nextGaps[i] = sortedAnglesDeg[(i + 1) % nProj] - sortedAnglesDeg[i];
    nextGaps[i] -= 360. * std::floor(nextGaps[i] / 360.);
  }
  const unsigned int largest = std::max_element(nextGaps.begin(), nextGaps.end()) - nextGaps.begin();

  std::vector<double> expected(nProj);
  for (unsigned int i = 0; i < nProj; i++)
  {
    double prevGap = nextGaps[(i + nProj - 1) % nProj];
    double nextGap = nextGaps[i];
    if (shortScan && i == (largest + 1) % nProj)
      prevGap = nextGap;
    if (shortScan && i == largest)
      nextGap = prevGap;
    expected[i] = 0.5 * (prevGap + nextGap) * itk::Math::pi_over_180;
  }

  // Shuffle the acquisition order: odd sorted positions first, then even ones
  std::vector<unsigned int> order;
  for (unsigned int i = 1; i < nProj; i += 2)
    order.push_back(i);
  for (unsigned int i = 0; i < nProj; i += 2)
    order.push_back(i);
  std::vector<double> anglesDeg;
  for (const unsigned int i : order)
    anglesDeg.push_back(sortedAnglesDeg[i]);

  auto geometry = CreateGeometry(anglesDeg);
  geometry->SetAngularGapThreshold(thresholdDeg * itk::Math::pi_over_180);
  const std::vector<double> gaps = geometry->GetAngularGaps(geometry->GetGantryAngles());

  bool ok = gaps.size() == nProj;
  for (unsigned int j = 0; ok && j < nProj; j++)
    ok &=
      CheckValue(testName, "angular gap at " + std::to_string(anglesDeg[j]) + " degrees", gaps[j], expected[order[j]]);
  ok &= CheckValue(testName,
                   "sum of angular gaps",
                   std::accumulate(gaps.begin(), gaps.end(), 0.),
                   std::accumulate(expected.begin(), expected.end(), 0.));
  return ok;
}

std::vector<double>
Range(double first, double last, double step)
{
  std::vector<double> angles;
  for (double a = first; a <= last + 1e-6; a += step)
    angles.push_back(a - 360. * std::floor(a / 360.));
  return angles;
}

std::vector<double>
Concatenate(std::vector<double> a, const std::vector<double> & b)
{
  a.insert(a.end(), b.begin(), b.end());
  return a;
}
} // namespace

int
rtkangulargapstest(int, char *[])
{
  bool ok = true;

  // Empty geometry
  {
    auto geometry = CreateGeometry({});
    if (!geometry->GetAngularGaps(geometry->GetGantryAngles()).empty())
    {
      std::cerr << "Empty geometry: angular gaps are not empty." << std::endl;
      ok = false;
    }
  }

  // Single projection
  {
    auto geometry = CreateGeometry({ 30. });
    ok &= CheckValue(
      "Single projection", "angular gap", geometry->GetAngularGaps(geometry->GetGantryAngles())[0], itk::Math::twopi);
  }

  // Full scans: regular, and with a gap below the angular gap threshold
  ok &= CheckAngularGaps("Full scan", Range(0., 359., 1.), false);
  ok &= CheckAngularGaps("Full scan with a gap", Range(0., 345., 1.), false);

  // Short scans: the first and last projections are weighted by the step to
  // their neighbor instead of half the largest gap
  ok &= CheckAngularGaps("Short scan", Range(0., 200., 1.), true);
  ok &= CheckAngularGaps("Short scan with a gap above the threshold", Range(0., 345., 1.), true, 10.);
  ok &= CheckAngularGaps("Short scan crossing 0", Range(300., 460., 2.), true);
  ok &= CheckAngularGaps("Short scan ending at 0", Range(160., 360., 2.), true);
  ok &= CheckAngularGaps(
    "Irregular short scan", Concatenate(Concatenate({ 10., 11. }, Range(13., 199., 2.)), { 204. }), true);
  ok &= CheckAngularGaps("Sparse short scan", { 0., 100., 200. }, true);

  // Short scan with repeated first and last angles: the projections with the
  // same angle share the weight of a single projection
  {
    std::vector<double>       anglesDeg = Concatenate(Concatenate({ 0. }, Range(0., 200., 1.)), { 200. });
    auto                      geometry = CreateGeometry(anglesDeg);
    const std::vector<double> gaps = geometry->GetAngularGaps(geometry->GetGantryAngles());
    const double              step = itk::Math::pi_over_180;
    const unsigned int        n = anglesDeg.size();
    ok &= CheckValue("Repeated angles", "sum of angular gaps at 0 degrees", gaps[0] + gaps[1], step);
    ok &= CheckValue("Repeated angles", "sum of angular gaps at 200 degrees", gaps[n - 2] + gaps[n - 1], step);
    for (unsigned int i = 2; i < n - 2; i++)
      ok &= CheckValue("Repeated angles", "angular gap " + std::to_string(i), gaps[i], step);
  }

  // Parallel geometry: the angles are taken modulo 180 degrees and the angles
  // which are redundant modulo 180 degrees share the weight of a single
  // projection
  {
    auto geometry = CreateGeometry(Range(0., 179., 1.), 0.);
    for (const double gap : geometry->GetAngularGaps(geometry->GetGantryAngles()))
      ok &= CheckValue("Parallel half scan", "angular gap", gap, itk::Math::pi_over_180);
  }
  {
    auto geometry = CreateGeometry(Range(0., 359., 1.), 0.);
    for (const double gap : geometry->GetAngularGaps(geometry->GetGantryAngles()))
      ok &= CheckValue("Parallel full scan", "angular gap", gap, 0.5 * itk::Math::pi_over_180);
  }
  {
    const std::vector<double> anglesDeg = Range(0., 199., 1.);
    auto                      geometry = CreateGeometry(anglesDeg, 0.);
    const std::vector<double> gaps = geometry->GetAngularGaps(geometry->GetGantryAngles());
    for (unsigned int i = 0; i < gaps.size(); i++)
    {
      const bool redundant = anglesDeg[i] < 20. || anglesDeg[i] >= 180.;
      ok &= CheckValue("Parallel scan over 200 degrees",
                       "angular gap at " + std::to_string(anglesDeg[i]) + " degrees",
                       gaps[i],
                       (redundant ? 0.5 : 1.) * itk::Math::pi_over_180);
    }
    ok &= CheckValue("Parallel scan over 200 degrees",
                     "sum of angular gaps",
                     std::accumulate(gaps.begin(), gaps.end(), 0.),
                     itk::Math::pi);
  }

  if (!ok)
    return EXIT_FAILURE;
  std::cout << "Test PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
