find . -maxdepth 1 -type f  ! -name "*.*"  -delete
find . -maxdepth 1 -type f  -name "*.o"  -delete
find . -maxdepth 1 -type f  -name "*.class"  -delete