#ifndef __LME_RABBITCT__H
#define __LME_RABBITCT__H 1

/** \brief RabbitCT global data structure.

  This is the main structure describing the relevant dataset descriptions.
  The notation is adapted to the MedicalPhysics Technical Note.
*/
struct RabbitCtGlobalData
  {
  ///@{ Relevant data for the backprojection
  unsigned int L;       ///< problem size in {128, 256, 512, 1024}
  unsigned int S_x;     ///< projection image width
  unsigned int S_y;     ///< projection image height (detector rows)
  double * A_n;         ///< 3x4 projetion matrix
  float * I_n;          ///< projection image buffer
  float R_L;            ///< isotropic voxel size
  float O_L;            ///< position of the 0-index in the world coordinate
                        // system
  float * f_L;          ///< pointer to where the result volume should be
                        // stored, by default managed by module.
  ///@}

  ///@{ Relevant data for projection image memory management. Only required for
  // advanced usage.
  unsigned int adv_numProjBuffers; ///< number of projection buffers in RAM
  float ** adv_pProjBuffers;       ///< projection image buffers, by default
                                   // managed by RabbitCT
  ///@}
  };

// the source files need to be compiled as a dynamically loadable shared
// library.
// In windows this is called DLL (dynamic link library) and in the linux world
// a shared library. For using this feature a special prefix is required for the
// function definitions. On windows systems it is given by
//    extern "C"__declspec(dllexport)
// and on linux systems by
//	extern "C"
#ifdef WIN32
#define FNCSIGN extern "C"__declspec (dllexport)
#elif WIN64
#define FNCSIGN extern "C"__declspec (dllexport)
#else
#define FNCSIGN extern "C"
#endif

#define RCT_FNCN_LOADALGORITHM      "RCTLoadAlgorithm"
#define RCT_FNCN_FINISHALGORITHM    "RCTFinishAlgorithm"
#define RCT_FNCN_UNLOADALGORITHM    "RCTUnloadAlgorithm"
#define RCT_FNCN_ALGORITHMBACKPROJ  "RCTAlgorithmBackprojection"

#endif
