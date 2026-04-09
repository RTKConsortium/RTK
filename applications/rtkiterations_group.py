import itk

__all__ = [
    "add_rtkiterations_group",
    "SetIterationsReportFromArgParse",
]


# Mimicks rtkiterations_section.ggo
def add_rtkiterations_group(parser):
    rtkiterations_group = parser.add_argument_group("Iteration reporting")
    rtkiterations_group.add_argument(
        "--output-every",
        help="Output intermediate reconstruction after some iterations",
        type=int,
    )
    rtkiterations_group.add_argument(
        "--iteration-file-name",
        help="File name to output intermediate iterations with {i} as a placeholder for iteration number",
    )


# Mimicks VerboseIterationCommand
class VerboseIterationCommand:
    def __init__(self):
        self.count = 0

    def callback(self):
        self.count = self.count + 1
        print(f"Iteration {self.count}", end="\r")


class VerboseEndCommand:
    def callback(self):
        print("")


# Mimicks OutputIterationCommand
class OutputIterationCommand:
    def __init__(self, reconstruction_filter, output_every, iteration_file_name):
        self.count = 0
        self.reconstruction_filter = reconstruction_filter
        self.output_every = output_every
        self.iteration_file_name = iteration_file_name

    def callback(self):
        self.count = self.count + 1
        if self.count % self.output_every == 0:
            output = self.reconstruction_filter.GetOutput()
            OutputImageType = type(output)
            if hasattr(output, "GetCudaDataManager"):
                OutputImageType = OutputImageType.__bases__[0]
            writer = itk.ImageFileWriter[OutputImageType].New()
            writer.SetInput(output)
            writer.SetFileName(self.iteration_file_name.format(i=self.count))
            writer.Update()


# Mimicks macro REPORT_ITERATIONS
def SetIterationsReportFromArgParse(args_info, filt):
    if args_info.verbose:
        cmd = VerboseIterationCommand()
        filt.AddObserver(itk.IterationEvent(), cmd.callback)
        cmd = VerboseEndCommand()
        filt.AddObserver(itk.EndEvent(), cmd.callback)
    if args_info.output_every is not None:
        cmd = OutputIterationCommand(
            filt, args_info.output_every, args_info.iteration_file_name
        )
        filt.AddObserver(itk.IterationEvent(), cmd.callback)
