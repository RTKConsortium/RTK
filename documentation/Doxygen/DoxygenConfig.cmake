# Configure CMake variables used to autogenerate a Doxyfile configuration.
#
# This file replaces the hand-maintained doxygen.config.in Doxyfile that RTK
# used previously. CMake's FindDoxygen module turns these DOXYGEN_* variables
# into a Doxyfile and provides the doxygen_add_docs() target. See
#   https://cmake.org/cmake/help/latest/module/FindDoxygen.html
#
# This mirrors the approach used by ITK.

set(DOXYGEN_PROJECT_NAME "RTK")
set(
  DOXYGEN_PROJECT_NUMBER
  "${RTK_VERSION_MAJOR}.${RTK_VERSION_MINOR}.${RTK_VERSION_PATCH}"
)
set(DOXYGEN_PROJECT_BRIEF "Reconstruction Toolkit")
# The logo is embedded by DoxygenHeader.html; doxygen cannot fetch a remote
# PROJECT_LOGO URL and would emit an error for it.
set(DOXYGEN_OUTPUT_DIRECTORY "${RTK_DOXYGEN_OUTPUT_DIR}")
set(DOXYGEN_BRIEF_MEMBER_DESC "NO")
set(DOXYGEN_FULL_PATH_NAMES "NO")
set(DOXYGEN_STRIP_FROM_PATH "${PROJECT_BINARY_DIR}/")
set(DOXYGEN_TAB_SIZE "2")
# Use ^^ instead of \n in aliases, see https://www.doxygen.nl/manual/custcmd.html
set(DOXYGEN_ALIASES "")
set(DOXYGEN_TOC_INCLUDE_HEADINGS "0")
set(DOXYGEN_BUILTIN_STL_SUPPORT "YES")
set(DOXYGEN_DISTRIBUTE_GROUP_DOC "YES")
set(DOXYGEN_LOOKUP_CACHE_SIZE "2")
set(DOXYGEN_CASE_SENSE_NAMES "YES")
set(DOXYGEN_EXTRACT_ALL "YES")
set(DOXYGEN_EXTRACT_PRIVATE "YES")
set(DOXYGEN_EXTRACT_STATIC "YES")
set(DOXYGEN_SORT_BRIEF_DOCS "YES")
set(DOXYGEN_ALPHABETICAL_INDEX "YES")
set(
  DOXYGEN_LAYOUT_FILE
  "${PROJECT_SOURCE_DIR}/documentation/Doxygen/DoxygenLayout.xml"
)
set(
  DOXYGEN_FILE_PATTERNS
  "*.h"
  "*.dox"
)
set(DOXYGEN_RECURSIVE "YES")
set(DOXYGEN_EXCLUDE "${PROJECT_SOURCE_DIR}/utilities")
set(DOXYGEN_EXCLUDE_PATTERNS "*_ggo.h")
set(DOXYGEN_EXCLUDE_SYMBOLS "itk::*")
set(DOXYGEN_EXAMPLE_PATTERNS "*.cxx")
set(DOXYGEN_EXAMPLE_RECURSIVE "YES")
set(DOXYGEN_INPUT_FILTER "${RTK_DOXYGEN_INPUT_FILTER}")
set(DOXYGEN_FILTER_SOURCE_FILES "YES")
set(DOXYGEN_SOURCE_BROWSER "YES")
set(DOXYGEN_REFERENCED_BY_RELATION "YES")
set(DOXYGEN_REFERENCES_RELATION "YES")
set(DOXYGEN_TEMPLATE_RELATIONS "YES")
set(
  DOXYGEN_HTML_HEADER
  "${PROJECT_SOURCE_DIR}/documentation/Doxygen/DoxygenHeader.html"
)
set(
  DOXYGEN_HTML_FOOTER
  "${PROJECT_SOURCE_DIR}/documentation/Doxygen/DoxygenFooter.html"
)
set(
  DOXYGEN_HTML_EXTRA_STYLESHEET
  "${PROJECT_SOURCE_DIR}/documentation/Doxygen/DoxygenStyle.css"
)
set(DOXYGEN_HTML_TIMESTAMP "YES")
# Doxygen 1.9 renamed HTML_TIMESTAMP to TIMESTAMP; CMake's FindDoxygen only
# writes options it knows for the detected doxygen version, so set both.
set(DOXYGEN_TIMESTAMP "YES")
set(DOXYGEN_HTML_DYNAMIC_SECTIONS "YES")
set(DOXYGEN_DOCSET_FEEDNAME "RTK Doxygen generated documentation")
set(DOXYGEN_DOCSET_BUNDLE_ID "org.rtk.RTK")
set(DOXYGEN_DOCSET_PUBLISHER_ID "org.rtk.RTKConsortium")
set(DOXYGEN_DOCSET_PUBLISHER_NAME "RTKConsortium")
set(DOXYGEN_ECLIPSE_DOC_ID "org.rtk.RTK")
set(DOXYGEN_ENUM_VALUES_PER_LINE "1")
set(DOXYGEN_USE_MATHJAX "YES")
set(
  DOXYGEN_MATHJAX_RELPATH
  "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/"
)
if(RTK_DOXYGEN_SERVER_BASED_SEARCH)
  set(DOXYGEN_SERVER_BASED_SEARCH "YES")
else()
  set(DOXYGEN_SERVER_BASED_SEARCH "NO")
endif()
set(DOXYGEN_LATEX_CMD_NAME "latex")
set(DOXYGEN_PAPER_TYPE "letter")
set(
  DOXYGEN_EXTRA_PACKAGES
  "amsmath"
  "amsfonts"
)
set(DOXYGEN_LATEX_BATCHMODE "YES")
set(DOXYGEN_ENABLE_PREPROCESSING "YES")
set(DOXYGEN_MACRO_EXPANSION "YES")
set(DOXYGEN_EXPAND_ONLY_PREDEF "YES")
set(DOXYGEN_SEARCH_INCLUDES "NO")
set(
  DOXYGEN_PREDEFINED
  "itkNotUsed(x)=x"
  "itkLegacyMacro(x)=[[deprecated]] x"
  "itkStaticConstMacro(name,type,value)=static constexpr type name = value"
  "itkSetInputMacro(name, type, number)=                   virtual void Set##name##Input(const type *_arg)\;                   virtual void SetInput##number(const type *_arg)\;"
  "itkGetInputMacro(name, type, number)=                   virtual const type * Get##name##Input() const\;                   virtual const type * GetInput##number() const\;"
  "itkSetMacro(name,type)=                   virtual void Set##name (type _arg)\;"
  "itkGetMacro(name,type)=                   virtual type Get##name ()\;"
  "itkGetConstMacro(name,type)=                   virtual type Get##name () const\;"
  "itkSetStringMacro(name)=                   virtual void Set##name (const char* _arg)\;"
  "itkGetStringMacro(name)=                   virtual const char* Get##name () const\;"
  "itkSetClampMacro(name,type,min,max)=                   virtual void Set##name (type _arg)\;"
  "itkSetObjectMacro(name,type)=                   virtual void Set##name (type* _arg)\;"
  "itkSetConstObjectMacro(name,type)=                   virtual void Set##name ( const type* _arg)\;"
  "itkGetConstObjectMacro(name,type)=      virtual const type* Get##name () const\;"
  "itkGetModifiableObjectMacro(name,type)= virtual type* GetModifiable##name ()\;  virtual const type* Get##name () const\;"
  "itkGetConstReferenceMacro(name,type)=                   virtual const type& Get##name () const\;"
  "itkSetEnumMacro(name,type)=                   virtual void Set##name (const type _arg)\;"
  "itkGetEnumMacro(name,type)=                   virtual type Get##name () const\;"
  "itkGetConstReferenceObjectMacro(name,type)=                   virtual const type::Pointer& Get##name () const\;"
  "itkSetDecoratedInputMacro(name, type)=                   virtual void Set##name##Input(const SimpleDataObjectDecorator<type> *_arg)\;                   virtual void Set##name(const type &_arg)\;              void Set##name(const SimpleDataObjectDecorator< type > *_arg)"
  "itkGetDecoratedInputMacro(name, type)= virtual const SimpleDataObjectDecorator< type > * Get##name##Input() const\; virtual const type & Get##name() const\;"
  "itkSetGetDecoratedInputMacro(name, type)=                   virtual void Set##name##Input(const SimpleDataObjectDecorator< type > *_arg)\;         virtual void Set##name(const type &_arg)\;          virtual void Set##name(const SimpleDataObjectDecorator< type > *_arg)\;        virtual const SimpleDataObjectDecorator< type > * Get##name##Input() const\;              virtual const type & Get##name() const\;"
  "itkSetDecoratedObjectInputMacro(name, type, number)=                   virtual void Set##name##Input(const DataObjectDecorator<type> *_arg)\;                   virtual void SetInput##number(const DataObjectDecorator<type> *_arg)\;                   virtual const DataObjectDecorator<type> * Get##name##Input() const\;                   virtual const DataObjectDecorator<type> * GetInput##number() const\;                   virtual void Set##name(const type *_arg)\;"
  "itkGetDecoratedObjectInputMacro(name, type)= virtual const DataObjectDecorator< type > * Get##name##Input() const\; virtual const type * Get##name() const\;"
  "itkSetGetDecoratedObjectInputMacro(name, type)= virtual void Set##name##Input(const DataObjectDecorator<type> *_arg)\;                   virtual void SetInput##number(const DataObjectDecorator<type> *_arg)\;                   virtual const DataObjectDecorator<type> * Get##name##Input() const\;                   virtual const DataObjectDecorator<type> * GetInput##number() const\;                   virtual void Set##name(const type *_arg)\; virtual const DataObjectDecorator< type > * Get##name##Input() const\; virtual const type * Get##name() const\;"
  "itkSetDecoratedOutputMacro(name, type)=                   virtual void Set##name##Output(const SimpleDataObjectDecorator<type> *_arg)\;                   virtual void Set##name(const type &_arg)\;              void Set##name(const SimpleDataObjectDecorator< type > *_arg)"
  "itkGetDecoratedOutputMacro(name, type)= virtual const SimpleDataObjectDecorator< type > * Get##name##Output() const\; virtual const type & Get##name() const\;"
  "itkBooleanMacro(name)=                   virtual void name##On ()\;                   virtual void name##Off ()\;"
  "itkSetVector2Macro(name,type)=                   virtual void Set##name (type _arg1, type _arg2)                   virtual void Set##name (type _arg[2])\;"
  "itkGetVector2Macro(name,type)=                   virtual type* Get##name () const\;                   virtual void Get##name (type& _arg1, type& _arg2) const\;                   virtual void Get##name (type _arg[2]) const\;"
  "itkSetVector3Macro(name,type)=                   virtual void Set##name (type _arg1, type _arg2, type _arg3)                   virtual void Set##name (type _arg[3])\;"
  "itkGetVector3Macro(name,type)=                   virtual type* Get##name () const\;                   virtual void Get##name (type& _arg1, type& _arg2, type& _arg3) const\;                   virtual void Get##name (type _arg[3]) const\;"
  "itkSetVector4Macro(name,type)=                   virtual void Set##name (type _arg1, type _arg2, type _arg3, type _arg4)                   virtual void Set##name (type _arg[4])\;"
  "itkGetVector4Macro(name,type)=                   virtual type* Get##name () const\;                   virtual void Get##name (type& _arg1, type& _arg2, type& _arg3, type& _arg4) const\;                   virtual void Get##name (type _arg[4]) const\;"
  "itkSetVector6Macro(name,type)=                   virtual void Set##name (type _arg1, type _arg2, type _arg3, type _arg4, type _arg5, type _arg6)                   virtual void Set##name (type _arg[6])\;"
  "itkGetVector6Macro(name,type)=                   virtual type* Get##name () const\;                   virtual void Get##name (type& _arg1, type& _arg2, type& _arg3, type& _arg4, type& _arg5, type& _arg6) const\;                   virtual void Get##name (type _arg[6]) const\;"
  "itkSetVectorMacro(name,type,count)=                   virtual void Set##name(type data[])\;"
  "itkGetVectorMacro(name,type,count)=                   virtual type* Get##name () const\;"
  "itkSetInputMacro(name,type)=         virtual void Set##name(const type *input)\;"
  "itkGetInputMacro(name,type)=         virtual const type * Get##name() const\;"
  "itkNewMacro(type)=                   static Pointer New()\;                                                                           virtual::itk::LightObject::Pointer CreateAnother() const\;"
  "itkSimpleNewMacro(type)=             static Pointer New()\;"
  "itkCreateAnotherMacro(type)=         virtual::itk::LightObject::Pointer CreateAnother() const\;"
  "itkFactorylessNewMacro(x)=                   static Pointer New()\;                   virtual ::itk::LightObject::Pointer CreateAnother() const\;"
  "itkOverrideGetNameOfClassMacro(thisClass)=                   virtual const char *GetNameOfClass() const\;"
  "itkVirtualGetNameOfClassMacro(thisClass)= virtual const char * GetNameOfClass() const\;"
  "itkOverrideGetNameOfClassMacro(thisClass)= const char * GetNameOfClass() const override\;"
  "itkEventMacro(thisClass,superclass)=                   class thisClass : public superclass {}\;"
  "itkDeclareExceptionMacro(newexcp,parentexcp,whatmessage)=         namespace itk {         class newexcp : public parentexcp {        public:       newexcp(const char *file, unsigned int lineNumber) :        parentexcp(file, lineNumber)          { this->SetDescription(whatmessage)\; }          newexcp(const std::string & file, unsigned int lineNumber) :          parentexcp(file, lineNumber)          { this->SetDescription(whatmessage)\; }          itkOverrideGetNameOfClassMacro(newexcp)\;         }\;        }"
  "itkConceptMacro(thisName,thisConcept)=                   /* This class requires thisName                       in the form of thisConcept */"
  "std::numeric_limits=                   std::numeric_limits"
  "ITK_TYPENAME=                   typename"
  "itkTemplateFloatingToIntegerMacro(name)=                   template <TReturn, typename TInput> name(TInput x)"
  "FEM_ABSTRACT_CLASS(thisClass,parentClass)=                   public:                                                                   /** Standard  Self typedef.*/                                          typedef thisClass Self\;                                                 /** Standard  Superclass typedef. */                                   typedef parentClass Superclass\;                                         /** Pointer or SmartPointer to an object. */                            typedef Self* Pointer\;                                                  /** Const pointer or SmartPointer to an object. */                      typedef const Self* ConstPointer\;                                     private:"
  "FEM_CLASS(thisClass,parentClass)=                   FEM_ABSTRACT_CLASS(thisClass,parentClass)                               public:                                                                   /** Create a new object from the existing one  */                       virtual Baseclass::Pointer Clone() const\;                               /** Class ID for FEM object factory */                                  static const int CLID\;                                                  /** Virtual function to access the class ID */                          virtual int ClassID() const                                               { return CLID\; }                                                      /** Object creation in an itk compatible way */                         static Self::Pointer New()                                                { return new Self()\; }                                              private:"
  "ERROR_CHECKING"
  "VCL_USE_NATIVE_STL=1"
  "VCL_USE_NATIVE_COMPLEX=1"
  "VCL_HAS_BOOL=1"
  "VXL_BIG_ENDIAN=1"
  "VXL_LITTLE_ENDIAN=0"
  "VNL_DLL_DATA="
  "size_t=vcl_size_t"
  "ITK_USE_FFTWD"
  "ITK_USE_FFTWF"
  "itkMacro_h"
  "ITK_LEGACY_REMOVE"
  "ITK_FUTURE_LEGACY_REMOVE"
  "ITKCommon_EXPORT_EXPLICIT"
  "ITK_OVERRIDE= override"
  "ITK_NULLPTR=  nullptr"
  "ITK_NOEXCEPT= noexcept"
  "ITK_DISALLOW_COPY_AND_MOVE(type)="
  "ITK_DISALLOW_COPY_AND_ASSIGN(type)="
  "ITK_FORCE_EXPORT_MACRO(X)="
)
# allow RTK_USE_CUDA code paths to be parsed when CUDA is not enabled
list(APPEND DOXYGEN_PREDEFINED "RTK_USE_CUDA")
set(DOXYGEN_GENERATE_TAGFILE "${RTK_DOXYGEN_OUTPUT_DIR}/RTKDoxygen.tag")
set(DOXYGEN_GRAPHICAL_HIERARCHY "NO")
set(DOXYGEN_DOT_IMAGE_FORMAT "svg")
set(DOXYGEN_DOT_GRAPH_MAX_NODES "150")
set(DOXYGEN_DOT_MULTI_TARGETS "YES")

# Map the RTK_DOXYGEN_<FORMAT> options on to doxygen's GENERATE_<FORMAT> tags.
# Doxygen's tag for compiled HTML help (.chm) is GENERATE_HTMLHELP, not
# GENERATE_CHM, so CHM maps to it specially.
foreach(
  _format
  IN
  ITEMS
    HTML
    DOCSET
    CHM
    QHP
    ECLIPSEHELP
    LATEX
    RTF
    XML
)
  if(_format STREQUAL "CHM")
    set(_doxygen_tag "HTMLHELP")
  else()
    set(_doxygen_tag "${_format}")
  endif()
  if(RTK_DOXYGEN_${_format})
    set(DOXYGEN_GENERATE_${_doxygen_tag} "YES")
  else()
    set(DOXYGEN_GENERATE_${_doxygen_tag} "NO")
  endif()
endforeach()
unset(_doxygen_tag)
unset(_format)
