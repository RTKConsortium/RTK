#ifndef __itkSheppLoganPhantomFilter_txx
#define __itkSheppLoganPhantomFilter_txx

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include "rtkHomogeneousMatrix.h"

namespace itk
{
template< class TInputImage, class TOutputImage >
void SheppLoganPhantomFilter< TInputImage, TOutputImage >::Config()
{
  const char *       search_fig = "Ellipsoid"; // Set search pattern
  int                offset = 0;
  std::string        line;
  std::ifstream      Myfile;

  Myfile.open( m_ConfigFile.c_str() );
  if ( !Myfile.is_open() )
    {
    itkGenericExceptionMacro("Error opening File");
    return;
    }

  while ( !Myfile.eof() )
    {
    getline(Myfile, line);
    if ( ( offset = line.find(search_fig, 0) ) != std::string::npos ) //Ellipsoid
                                                                      // found
      {
      const std::string parameterNames[8] = { "x", "y", "z", "A", "B", "C", "beta", "gray" };
      std::vector<double> parameters;
      for ( int j = 0; j < 8; j++ )
        {
        double val = 0.;
        if ( ( offset = line.find(parameterNames[j], 0) ) != std::string::npos )
          {
          offset += parameterNames[j].length()+1;
          std::string s = line.substr(offset,line.length()-offset);
          std::istringstream ss(s);
          ss >> val;
          //Saving all parameters for each ellipsoid
          }
        parameters.push_back(val);
        }
      m_Fig.push_back(parameters);
      }
    }
  Myfile.close();
}


template< class TInputImage, class TOutputImage >
void SheppLoganPhantomFilter< TInputImage, TOutputImage >::GenerateData()
{
  std::cout << "Configuration Process...\n" << std::endl;
  this->Config();

  std::vector< REIType::Pointer > rei( m_Fig.size() );
  for ( unsigned int i = 0; i < m_Fig.size(); i++ )
    {
    rei[i] = REIType::New();

    rei[i]->SetMultiplicativeConstant(m_Fig[i][7]); //Set GrayScale value
    rei[i]->SetSemiPrincipalAxisX(m_Fig[i][0]);
    rei[i]->SetSemiPrincipalAxisY(m_Fig[i][1]);
    rei[i]->SetSemiPrincipalAxisZ(m_Fig[i][2]);

    rei[i]->SetCenterX(m_Fig[i][3]);
    rei[i]->SetCenterY(m_Fig[i][4]);
    rei[i]->SetCenterZ(m_Fig[i][5]);

    rei[i]->SetRotationAngle(m_Fig[i][6]);

    if ( i == ( m_Fig.size() - 1 ) ) //last case
      {
      if(i==0) //just one ellipsoid
        rei[i]->SetInput( rei[i]->GetOutput() );
      else
        rei[i]->SetInput( rei[i-1]->GetOutput() );
      rei[i]->SetGeometry( this->GetGeometry() );
      }

    if (i>0) //other cases
      {
      rei[i]->SetInput( rei[i-1]->GetOutput() );
      rei[i]->SetGeometry( this->GetGeometry() );
      }

    else //first case
      {
      rei[i]->SetInput( this->GetInput() );
      rei[i]->SetGeometry( this->GetGeometry() );
      }
    }
//Update
  rei[ m_Fig.size() - 1]->Update();
  this->GraftOutput( rei[m_Fig.size()-1]->GetOutput() );
}
} // end namespace itk

#endif
