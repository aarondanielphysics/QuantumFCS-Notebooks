(* Lightweight preflight used by the parameter-search notebook. *)
projectRootFromEnv = Environment["FCS_PROJECT_ROOT"];
projectRoot = If[
  projectRootFromEnv === $Failed || StringTrim[projectRootFromEnv] === "",
  DirectoryName[DirectoryName[$InputFileName]],
  projectRootFromEnv
];
meltLocal = FileNameJoin[{projectRoot, "melt.m"}];

Print["Checking MELT load"];
Print["  InputFileName: ", ToString[$InputFileName, InputForm]];
Print["  ScriptCommandLine: ", ToString[$ScriptCommandLine, InputForm]];
Print["  FCS_PROJECT_ROOT: ", ToString[projectRootFromEnv, InputForm]];
Print["  projectRoot: ", ToString[projectRoot, InputForm]];
Print["  melt.m: ", ToString[meltLocal, InputForm]];

If[! FileExistsQ[meltLocal],
  Print["MELT file is missing at: ", meltLocal];
  Quit[2]
];

Check[
  Get[meltLocal],
  Print["Failed to Get local MELT file: ", ToString[meltLocal, InputForm]];
  Quit[3]
];

If[
  ! And @@ (NameQ /@ {"FCSAverage", "FCSNoise", "Liouvillian", "SteadyState", "LoadBosonicOperators"}),
  Print["MELT loaded, but required symbols are not available."];
  Quit[4]
];

Print["MELT load check passed."];
