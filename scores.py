gpt4o_scores = [('Html', 32.0), ('MD', 32.0), ('Scala$', 32.0), ('VimL', 40.0), ('Erlang', 44.0),
                ('Pascal', 52.0), ('AWK', 54.0), ('Dart', 54.9), ('TS', 56.0), ('C++', 58.0),
                ('C', 60.0), ('Clip', 60.0), ('Lua', 60.0), ('Go', 62.0), ('JS', 62.0),
                ('Elispi', 64.0), ('Perl', 64.0), ('PHP', 64.0), ('Ruby', 64.0), ('Elixir', 66.0),
                ('Fortran', 66.0), ('R', 66.0), ('Racket', 66.0), ('Tcl', 68.0), ('C#', 72.0),
                ('Julia', 72.0), ('Power', 72.0), ('Json', 74.0), ('Python$', 76.0), ('Scheme', 76.0),
                ('Shell', 76.0), ('F#', 78.0), ('VB', 78.0), ('Groovy', 80.0), ('Java', 81.1),
                ('Coffee', 82.0), ('Rust', 83.0), ('Kotlin', 84.0), ('Swift', 84.0), ('Haskell', 90.0)]


gpt4o_scores.sort(key=lambda x: x[1])
print(gpt4o_scores)