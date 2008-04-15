# QFITS CHECK MEMORY  
#--------------------
AC_DEFUN([QFITS_CHECK_MEMORY],
[
    AC_MSG_CHECKING([qfits options])

    AC_ARG_ENABLE(memory-mode,
                AC_HELP_STRING([--enable-memory-mode=M],
                               [where M=0 switches off the internal memory
                                handling, M=1 exits the program
                                whenever a memory allocation fails,
                                M=2 switches on the internal memory
                                handling, M=3 switches on the internal memory
                                handling and the memory debug mode]),
                [
                    qfits_memory_flag=yes
                    # $enableval=yes when no argument is given
                    qfits_memory_mode=$enableval
                ])

    AC_ARG_ENABLE(max-ptrs,
                AC_HELP_STRING([--enable-max-ptrs=MAXPTRS],
                               [MAXPTRS Set MAXPTRS as the maximum number of
				pointers allowed]),
                [
                    qfits_max_ptrs_flag=yes
                    qfits_max_ptrs=$enableval
                ])

    # Pending: check qfits_max_ptrs is numeric, otherwise AC_MSG_ERROR 
    if test "x$qfits_max_ptrs_flag" = xyes ; then
        QFITS_MAXPTRS_CFLAGS="-DQFITS_MEMORY_MAXPTRS=$qfits_max_ptrs"
    else
        QFITS_MAXPTRS_CFLAGS=""
    fi

    if test "x$qfits_memory_flag" = xyes ; then
        QFITS_CFLAGS="-DQFITS_MEMORY_MODE=$qfits_memory_mode"
        case $qfits_memory_mode in
        yes)        
          QFITS_CFLAGS="-DQFITS_MEMORY_MODE=0 -DQFITS_MEMORY_MAXPRS=1" 
          break ;;
        0|1)        
          QFITS_CFLAGS="-DQFITS_MEMORY_MODE=$qfits_memory_mode -DQFITS_MEMORY_MAXPRS=1" 
          break ;;
        2)        
          QFITS_CFLAGS="-DQFITS_MEMORY_MODE=2 $QFITS_MAXPTRS_CFLAGS" 
          break ;;
        3)        
          QFITS_CFLAGS="-DQFITS_MEMORY_MODE=2 -DQFITS_MEMORY_DEBUG=2 $QFITS_MAXPTRS_CFLAGS" 
          break ;;
        *)
          AC_MSG_ERROR([Option --enable-memory-mode=$qfits_memory_mode not valid. Please check!])
          break ;;
        esac

    else
        QFITS_CFLAGS="$QFITS_MAXPTRS_CFLAGS" 
    fi

    AC_MSG_RESULT([QFITS_CFLAGS=$QFITS_CFLAGS])
    AC_SUBST(QFITS_CFLAGS)
])
