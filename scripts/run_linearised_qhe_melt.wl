(* The parameter-search notebook passes this explicitly so paths with spaces are stable. *)
projectRootFromEnv = Environment["FCS_PROJECT_ROOT"];
projectRoot = If[
  projectRootFromEnv === $Failed || StringTrim[projectRootFromEnv] === "",
  DirectoryName[DirectoryName[$InputFileName]],
  projectRootFromEnv
];
rawDataDir = FileNameJoin[{projectRoot, "data"}];
If[! DirectoryQ[rawDataDir], CreateDirectory[rawDataDir]];

LoadMELT[] := Module[{meltLocal},
  meltLocal = FileNameJoin[{projectRoot, "melt.m"}];
  Print["Loading MELT"];
  Print["  InputFileName: ", ToString[$InputFileName, InputForm]];
  Print["  ScriptCommandLine: ", ToString[$ScriptCommandLine, InputForm]];
  Print["  FCS_PROJECT_ROOT: ", ToString[projectRootFromEnv, InputForm]];
  Print["  melt.m: ", ToString[meltLocal, InputForm]];
  If[FileExistsQ[meltLocal],
    Check[
      Get[meltLocal],
      Print["Failed to Get local MELT file: ", ToString[meltLocal, InputForm]];
      Quit[2]
    ],
    If[Environment["MELT_ALLOW_DOWNLOAD"] === "1",
      Check[
        Get["http://www.fmt.if.usp.br/~gtlandi/download/melt.m"],
        Print["Failed to download and load MELT from the upstream URL."];
        Quit[2]
      ],
      Print["MELT did not load. Place melt.m at the repository root or rerun with MELT_ALLOW_DOWNLOAD=1."];
      Quit[1]
    ]
  ];
  If[
    ! And @@ (NameQ /@ {"FCSAverage", "FCSNoise", "Liouvillian", "SteadyState", "LoadBosonicOperators"}),
    Print["MELT did not load. Place melt.m at the repository root or enable access to the MELT URL."];
    Quit[1]
  ];
];

ParseIntegerListEnv[name_, default_] := Module[{value},
  value = Environment[name];
  If[value === $Failed || StringTrim[value] === "",
    default,
    ToExpression /@ StringSplit[StringReplace[value, ";" -> ","], ","]
  ]
];

ParsePositiveIntegerEnv[name_, default_] := Module[{value, parsed},
  value = Environment[name];
  If[value === $Failed || StringTrim[value] === "", Return[default]];
  parsed = ToExpression[value];
  If[! IntegerQ[parsed] || parsed < 1,
    Print["Expected a positive integer for ", name, ", got: ", value];
    Quit[3]
  ];
  parsed
];

LoadMELT[];

(* Keep the script defaults aligned with the Julia benchmark parameters. *)
g = ToExpression[Environment["MELT_LINEARISED_G"] /. $Failed -> "0.35"];
kappa = ToExpression[Environment["MELT_LINEARISED_KAPPA"] /. $Failed -> "1.0"];
kappaH = kappa;
kappaC = kappa;
nh = ToExpression[Environment["MELT_LINEARISED_NH"] /. $Failed -> "0.5"];
nc = ToExpression[Environment["MELT_LINEARISED_NC"] /. $Failed -> "0.05"];

BuildLinearisedModel[dimH_Integer, dimC_Integer, gVal_] := Module[
  {aH, adH, aC, adC, idH, idC, aHfull, aCfull, adHfull, adCfull, H, jumps, L, rhoSS},
  LoadBosonicOperators[dimH, 1.0];
  aH = a;
  adH = ConjugateTranspose[aH];
  idH = IdentityMatrix[dimH];

  LoadBosonicOperators[dimC, 1.0];
  aC = a;
  adC = ConjugateTranspose[aC];
  idC = IdentityMatrix[dimC];

  aHfull = KroneckerProduct[aH, idC];
  aCfull = KroneckerProduct[idH, aC];
  adHfull = KroneckerProduct[adH, idC];
  adCfull = KroneckerProduct[idH, adC];

  H = gVal*(adCfull.aHfull + adHfull.aCfull);
  jumps = {
    Sqrt[(nh + 1)*kappaH]*aHfull,
    Sqrt[(nc + 1)*kappaC]*aCfull,
    Sqrt[nh*kappaH]*adHfull,
    Sqrt[nc*kappaC]*adCfull
  };
  L = Liouvillian[H, jumps];
  rhoSS = SteadyState[L];
  {H, jumps, L, rhoSS}
];

dims = ParseIntegerListEnv["MELT_LINEARISED_DIMS", Range[2, 9]];
timingSamples = ParsePositiveIntegerEnv["MELT_LINEARISED_SAMPLES", 5];
Print["Running MELT linearised QHE benchmark"];
Print["Local dimensions: ", dims];
Print["Mean timing samples per dimension: ", timingSamples];

timings = Table[
  Module[{Hmod, jumpsMod, Lmod, rhoMod, sampleTimings},
    Print["Benchmarking local dimension d = ", d, ", Hilbert dimension = ", d*d];
    {Hmod, jumpsMod, Lmod, rhoMod} = BuildLinearisedModel[d, d, g];
    (* Repeat only the FCS timing block and export the arithmetic mean. *)
    sampleTimings = Table[
      First @ AbsoluteTiming[
        {
          FCSAverage[rhoMod, Lmod, {jumpsMod[[2]], jumpsMod[[4]]}, {-1, 1}, "jump"],
          FCSNoise[rhoMod, Lmod, {jumpsMod[[2]], jumpsMod[[4]]}, {-1, 1}, "jump"]
        }
      ],
      {sample, timingSamples}
    ];
    Mean[sampleTimings]
  ],
  {d, dims}
];

outputFile = FileNameJoin[{rawDataDir, "benchmark_melt_linearised_vs_dimension.csv"}];
Export[outputFile, Transpose[{dims*dims, timings*10^6}], "CSV"];
Print["Saved: ", outputFile];
