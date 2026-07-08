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

omegaH = ToExpression[Environment["MELT_DENSE_OMEGA_H"] /. $Failed -> "5.0"];
omegaC = ToExpression[Environment["MELT_DENSE_OMEGA_C"] /. $Failed -> "1.0"];
(* Keep the script defaults aligned with the Julia benchmark parameters. *)
EJ = ToExpression[Environment["MELT_DENSE_EJ"] /. $Failed -> "1.75"];
lambdaH = ToExpression[Environment["MELT_DENSE_LAMBDA_H"] /. $Failed -> "0.20"];
lambdaC = ToExpression[Environment["MELT_DENSE_LAMBDA_C"] /. $Failed -> "0.25"];
kappaH = ToExpression[Environment["MELT_DENSE_KAPPA_H"] /. $Failed -> "1.0"];
kappaC = ToExpression[Environment["MELT_DENSE_KAPPA_C"] /. $Failed -> "1.0"];
nbarH = ToExpression[Environment["MELT_DENSE_NBAR_H"] /. $Failed -> "0.50"];
nbarC = ToExpression[Environment["MELT_DENSE_NBAR_C"] /. $Failed -> "0.05"];

BuildDenseCircuitQHEModel[dimH_Integer, dimC_Integer] := Module[
  {
    aH, adH, aC, adC, idH, idC, aHfull, adHfull, aCfull, adCfull,
    phi, exp2iPhi, cos2Phi, H, jumps, L, rhoSS
  },
  LoadBosonicOperators[dimH, 1.0];
  aH = a;
  adH = ConjugateTranspose[aH];
  idH = IdentityMatrix[dimH];

  LoadBosonicOperators[dimC, 1.0];
  aC = a;
  adC = ConjugateTranspose[aC];
  idC = IdentityMatrix[dimC];

  aHfull = KroneckerProduct[aH, idC];
  adHfull = KroneckerProduct[adH, idC];
  aCfull = KroneckerProduct[idH, aC];
  adCfull = KroneckerProduct[idH, adC];

  phi = lambdaH*(adHfull + aHfull) + lambdaC*(adCfull + aCfull);
  exp2iPhi = MatrixExp[2*I*phi];
  cos2Phi = (exp2iPhi + ConjugateTranspose[exp2iPhi])/2;

  H = omegaH*(adHfull.aHfull) + omegaC*(adCfull.aCfull) - EJ*cos2Phi;
  jumps = {
    Sqrt[kappaH*(nbarH + 1)]*aHfull,
    Sqrt[kappaC*(nbarC + 1)]*aCfull,
    Sqrt[kappaH*nbarH]*adHfull,
    Sqrt[kappaC*nbarC]*adCfull
  };
  L = Liouvillian[H, jumps];
  rhoSS = SteadyState[L];

  <|
    "HilbertDimension" -> dimH*dimC,
    "H" -> H,
    "Jumps" -> jumps,
    "L" -> L,
    "RhoSS" -> rhoSS,
    "MonitoredJumps" -> {jumps[[1]], jumps[[3]]},
    "Weights" -> {-1, 1}
  |>
];

localDimensions = ParseIntegerListEnv["MELT_DENSE_DIMS", Range[2, 8]];
timingSamples = ParsePositiveIntegerEnv["MELT_DENSE_SAMPLES", 10];
Print["Running MELT dense circuit-QED QHE benchmark"];
Print["Local dimensions: ", localDimensions];
Print["Mean timing samples per dimension: ", timingSamples];

timings = Table[
  Module[{model, sampleTimings},
    Print["Benchmarking local dimension d = ", d, ", Hilbert dimension = ", d*d];
    model = BuildDenseCircuitQHEModel[d, d];
    (* Repeat only the FCS timing block and export the arithmetic mean. *)
    sampleTimings = Table[
      First @ AbsoluteTiming[
        {
          FCSAverage[
            model["RhoSS"], model["L"], model["MonitoredJumps"], model["Weights"], "jump"
          ],
          FCSNoise[
            model["RhoSS"], model["L"], model["MonitoredJumps"], model["Weights"], "jump"
          ]
        }
      ],
      {sample, timingSamples}
    ];
    Mean[sampleTimings]
  ],
  {d, localDimensions}
];

outputFile = FileNameJoin[{rawDataDir, "benchmark_melt_dense_circuit_qhe_vs_dimension.csv"}];
Export[outputFile, Transpose[{localDimensions*localDimensions, timings*10^6}], "CSV"];
Print["Saved: ", outputFile];
