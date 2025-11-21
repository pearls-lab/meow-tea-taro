/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output, and Bison version.  */
#define YYBISON 30802

/* Bison version string.  */
#define YYBISON_VERSION "3.8.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1


/* Substitute the variable and function names.  */
#define yyparse         ops_pddlparse
#define yylex           ops_pddllex
#define yyerror         ops_pddlerror
#define yydebug         ops_pddldebug
#define yynerrs         ops_pddlnerrs
#define yylval          ops_pddllval
#define yychar          ops_pddlchar

/* First part of user prologue.  */
#line 1 "scan-ops_pddl.y"

#ifdef YYDEBUG
  extern int yydebug=1;
#endif


#include <stdio.h>
#include <string.h> 
#include "ff.h"
#include "memory.h"
#include "parse.h"

#ifndef YYMAXDEPTH
#define YYMAXDEPTH 10000000
#endif


#ifndef SCAN_ERR
#define SCAN_ERR
#define DOMDEF_EXPECTED            0
#define DOMAIN_EXPECTED            1
#define DOMNAME_EXPECTED           2
#define LBRACKET_EXPECTED          3
#define RBRACKET_EXPECTED          4
#define DOMDEFS_EXPECTED           5
#define REQUIREM_EXPECTED          6
#define TYPEDLIST_EXPECTED         7
#define LITERAL_EXPECTED           8
#define PRECONDDEF_UNCORRECT       9
#define TYPEDEF_EXPECTED          10
#define CONSTLIST_EXPECTED        11
#define PREDDEF_EXPECTED          12 
#define NAME_EXPECTED             13
#define VARIABLE_EXPECTED         14
#define ACTIONFUNCTOR_EXPECTED    15
#define ATOM_FORMULA_EXPECTED     16
#define EFFECT_DEF_EXPECTED       17
#define NEG_FORMULA_EXPECTED      18
#define NOT_SUPPORTED             19
#define ACTION                    20
#endif


#define NAME_STR "name\0"
#define VARIABLE_STR "variable\0"
#define STANDARD_TYPE "OBJECT\0"
 

static char *serrmsg[] = {
  "domain definition expected",
  "'domain' expected",
  "domain name expected",
  "'(' expected",
  "')' expected",
  "additional domain definitions expected",
  "requirements (e.g. ':STRIPS') expected",
  "typed list of <%s> expected",
  "literal expected",
  "uncorrect precondition definition",
  "type definition expected",
  "list of constants expected",
  "predicate definition expected",
  "<name> expected",
  "<variable> expected",
  "action functor expected",
  "atomic formula expected",
  "effect definition expected",
  "negated atomic formula expected",
  "requirement %s not supported by this FF version",  
  "action definition is not correct",
  NULL
};


/* void opserr( int errno, char *par ); */
void opserr( int errno, char *par )

{

/*   sact_err = errno; */

/*   if ( sact_err_par ) { */
/*     free(sact_err_par); */
/*   } */
/*   if ( par ) { */
/*     sact_err_par = new_Token(strlen(par)+1); */
/*     strcpy(sact_err_par, par); */
/*   } else { */
/*     sact_err_par = NULL; */
/*   } */

}


static int sact_err;
static char *sact_err_par = NULL;
static PlOperator *scur_op = NULL;
static Bool sis_negated = FALSE;


int supported( char *str )

{

  int i;
  char * sup[] = { ":STRIPS", ":NEGATION", ":EQUALITY",":TYPING", 
		   ":CONDITIONAL-EFFECTS", ":NEGATIVE-PRECONDITIONS", ":DISJUNCTIVE-PRECONDITIONS", 
		   ":EXISTENTIAL-PRECONDITIONS", ":UNIVERSAL-PRECONDITIONS", 
		   ":QUANTIFIED-PRECONDITIONS", ":ADL", ":FLUENTS", ":ACTION-COSTS", ":DERIVED-PREDICATES",
		   NULL };     

  for (i=0; NULL != sup[i]; i++) {
    if ( SAME == strcmp(sup[i], str) ) {
      return TRUE;
    }
  }
  
  return FALSE;

}


#line 201 "scan-ops_pddl.tab.c"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif


/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int ops_pddldebug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    DEFINE_TOK = 258,              /* DEFINE_TOK  */
    DOMAIN_TOK = 259,              /* DOMAIN_TOK  */
    REQUIREMENTS_TOK = 260,        /* REQUIREMENTS_TOK  */
    TYPES_TOK = 261,               /* TYPES_TOK  */
    NUMBER_TOK = 262,              /* NUMBER_TOK  */
    EITHER_TOK = 263,              /* EITHER_TOK  */
    CONSTANTS_TOK = 264,           /* CONSTANTS_TOK  */
    PREDICATES_TOK = 265,          /* PREDICATES_TOK  */
    FUNCTIONS_TOK = 266,           /* FUNCTIONS_TOK  */
    ACTION_TOK = 267,              /* ACTION_TOK  */
    AXIOM_TOK = 268,               /* AXIOM_TOK  */
    VARS_TOK = 269,                /* VARS_TOK  */
    IMPLIES_TOK = 270,             /* IMPLIES_TOK  */
    PRECONDITION_TOK = 271,        /* PRECONDITION_TOK  */
    PARAMETERS_TOK = 272,          /* PARAMETERS_TOK  */
    EFFECT_TOK = 273,              /* EFFECT_TOK  */
    AND_TOK = 274,                 /* AND_TOK  */
    NOT_TOK = 275,                 /* NOT_TOK  */
    WHEN_TOK = 276,                /* WHEN_TOK  */
    FORALL_TOK = 277,              /* FORALL_TOK  */
    IMPLY_TOK = 278,               /* IMPLY_TOK  */
    OR_TOK = 279,                  /* OR_TOK  */
    EXISTS_TOK = 280,              /* EXISTS_TOK  */
    LE_TOK = 281,                  /* LE_TOK  */
    LEQ_TOK = 282,                 /* LEQ_TOK  */
    EQ_TOK = 283,                  /* EQ_TOK  */
    GEQ_TOK = 284,                 /* GEQ_TOK  */
    GE_TOK = 285,                  /* GE_TOK  */
    MINUS_TOK = 286,               /* MINUS_TOK  */
    AD_TOK = 287,                  /* AD_TOK  */
    MU_TOK = 288,                  /* MU_TOK  */
    DI_TOK = 289,                  /* DI_TOK  */
    ASSIGN_TOK = 290,              /* ASSIGN_TOK  */
    SCALE_UP_TOK = 291,            /* SCALE_UP_TOK  */
    SCALE_DOWN_TOK = 292,          /* SCALE_DOWN_TOK  */
    INCREASE_TOK = 293,            /* INCREASE_TOK  */
    DECREASE_TOK = 294,            /* DECREASE_TOK  */
    NAME = 295,                    /* NAME  */
    VARIABLE = 296,                /* VARIABLE  */
    NUM = 297,                     /* NUM  */
    OPEN_PAREN = 298,              /* OPEN_PAREN  */
    CLOSE_PAREN = 299              /* CLOSE_PAREN  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 128 "scan-ops_pddl.y"


  char string[MAX_LENGTH];
  char *pstring;
  ParseExpNode *pParseExpNode;
  PlNode *pPlNode;
  FactList *pFactList;
  TokenList *pTokenList;
  TypedList *pTypedList;


#line 304 "scan-ops_pddl.tab.c"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE ops_pddllval;


int ops_pddlparse (void);



/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_DEFINE_TOK = 3,                 /* DEFINE_TOK  */
  YYSYMBOL_DOMAIN_TOK = 4,                 /* DOMAIN_TOK  */
  YYSYMBOL_REQUIREMENTS_TOK = 5,           /* REQUIREMENTS_TOK  */
  YYSYMBOL_TYPES_TOK = 6,                  /* TYPES_TOK  */
  YYSYMBOL_NUMBER_TOK = 7,                 /* NUMBER_TOK  */
  YYSYMBOL_EITHER_TOK = 8,                 /* EITHER_TOK  */
  YYSYMBOL_CONSTANTS_TOK = 9,              /* CONSTANTS_TOK  */
  YYSYMBOL_PREDICATES_TOK = 10,            /* PREDICATES_TOK  */
  YYSYMBOL_FUNCTIONS_TOK = 11,             /* FUNCTIONS_TOK  */
  YYSYMBOL_ACTION_TOK = 12,                /* ACTION_TOK  */
  YYSYMBOL_AXIOM_TOK = 13,                 /* AXIOM_TOK  */
  YYSYMBOL_VARS_TOK = 14,                  /* VARS_TOK  */
  YYSYMBOL_IMPLIES_TOK = 15,               /* IMPLIES_TOK  */
  YYSYMBOL_PRECONDITION_TOK = 16,          /* PRECONDITION_TOK  */
  YYSYMBOL_PARAMETERS_TOK = 17,            /* PARAMETERS_TOK  */
  YYSYMBOL_EFFECT_TOK = 18,                /* EFFECT_TOK  */
  YYSYMBOL_AND_TOK = 19,                   /* AND_TOK  */
  YYSYMBOL_NOT_TOK = 20,                   /* NOT_TOK  */
  YYSYMBOL_WHEN_TOK = 21,                  /* WHEN_TOK  */
  YYSYMBOL_FORALL_TOK = 22,                /* FORALL_TOK  */
  YYSYMBOL_IMPLY_TOK = 23,                 /* IMPLY_TOK  */
  YYSYMBOL_OR_TOK = 24,                    /* OR_TOK  */
  YYSYMBOL_EXISTS_TOK = 25,                /* EXISTS_TOK  */
  YYSYMBOL_LE_TOK = 26,                    /* LE_TOK  */
  YYSYMBOL_LEQ_TOK = 27,                   /* LEQ_TOK  */
  YYSYMBOL_EQ_TOK = 28,                    /* EQ_TOK  */
  YYSYMBOL_GEQ_TOK = 29,                   /* GEQ_TOK  */
  YYSYMBOL_GE_TOK = 30,                    /* GE_TOK  */
  YYSYMBOL_MINUS_TOK = 31,                 /* MINUS_TOK  */
  YYSYMBOL_AD_TOK = 32,                    /* AD_TOK  */
  YYSYMBOL_MU_TOK = 33,                    /* MU_TOK  */
  YYSYMBOL_DI_TOK = 34,                    /* DI_TOK  */
  YYSYMBOL_ASSIGN_TOK = 35,                /* ASSIGN_TOK  */
  YYSYMBOL_SCALE_UP_TOK = 36,              /* SCALE_UP_TOK  */
  YYSYMBOL_SCALE_DOWN_TOK = 37,            /* SCALE_DOWN_TOK  */
  YYSYMBOL_INCREASE_TOK = 38,              /* INCREASE_TOK  */
  YYSYMBOL_DECREASE_TOK = 39,              /* DECREASE_TOK  */
  YYSYMBOL_NAME = 40,                      /* NAME  */
  YYSYMBOL_VARIABLE = 41,                  /* VARIABLE  */
  YYSYMBOL_NUM = 42,                       /* NUM  */
  YYSYMBOL_OPEN_PAREN = 43,                /* OPEN_PAREN  */
  YYSYMBOL_CLOSE_PAREN = 44,               /* CLOSE_PAREN  */
  YYSYMBOL_YYACCEPT = 45,                  /* $accept  */
  YYSYMBOL_file = 46,                      /* file  */
  YYSYMBOL_47_1 = 47,                      /* $@1  */
  YYSYMBOL_domain_definition = 48,         /* domain_definition  */
  YYSYMBOL_49_2 = 49,                      /* $@2  */
  YYSYMBOL_domain_name = 50,               /* domain_name  */
  YYSYMBOL_optional_domain_defs = 51,      /* optional_domain_defs  */
  YYSYMBOL_predicates_def = 52,            /* predicates_def  */
  YYSYMBOL_53_3 = 53,                      /* $@3  */
  YYSYMBOL_predicates_list = 54,           /* predicates_list  */
  YYSYMBOL_55_4 = 55,                      /* $@4  */
  YYSYMBOL_functions_def = 56,             /* functions_def  */
  YYSYMBOL_57_5 = 57,                      /* $@5  */
  YYSYMBOL_functions_list = 58,            /* functions_list  */
  YYSYMBOL_59_6 = 59,                      /* $@6  */
  YYSYMBOL_60_7 = 60,                      /* $@7  */
  YYSYMBOL_require_def = 61,               /* require_def  */
  YYSYMBOL_62_8 = 62,                      /* $@8  */
  YYSYMBOL_63_9 = 63,                      /* $@9  */
  YYSYMBOL_require_key_star = 64,          /* require_key_star  */
  YYSYMBOL_65_10 = 65,                     /* $@10  */
  YYSYMBOL_types_def = 66,                 /* types_def  */
  YYSYMBOL_67_11 = 67,                     /* $@11  */
  YYSYMBOL_constants_def = 68,             /* constants_def  */
  YYSYMBOL_69_12 = 69,                     /* $@12  */
  YYSYMBOL_action_def = 70,                /* action_def  */
  YYSYMBOL_71_13 = 71,                     /* $@13  */
  YYSYMBOL_72_14 = 72,                     /* $@14  */
  YYSYMBOL_axiom_def = 73,                 /* axiom_def  */
  YYSYMBOL_param_def = 74,                 /* param_def  */
  YYSYMBOL_action_def_body = 75,           /* action_def_body  */
  YYSYMBOL_76_15 = 76,                     /* $@15  */
  YYSYMBOL_77_16 = 77,                     /* $@16  */
  YYSYMBOL_adl_goal_description = 78,      /* adl_goal_description  */
  YYSYMBOL_adl_goal_description_star = 79, /* adl_goal_description_star  */
  YYSYMBOL_adl_effect = 80,                /* adl_effect  */
  YYSYMBOL_adl_effect_star = 81,           /* adl_effect_star  */
  YYSYMBOL_f_head = 82,                    /* f_head  */
  YYSYMBOL_f_exp = 83,                     /* f_exp  */
  YYSYMBOL_literal_term = 84,              /* literal_term  */
  YYSYMBOL_atomic_formula_term = 85,       /* atomic_formula_term  */
  YYSYMBOL_term_star = 86,                 /* term_star  */
  YYSYMBOL_term = 87,                      /* term  */
  YYSYMBOL_name_plus = 88,                 /* name_plus  */
  YYSYMBOL_predicate = 89,                 /* predicate  */
  YYSYMBOL_typed_list_name = 90,           /* typed_list_name  */
  YYSYMBOL_typed_list_variable = 91        /* typed_list_variable  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;




#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

/* Work around bug in HP-UX 11.23, which defines these macros
   incorrectly for preprocessor constants.  This workaround can likely
   be removed in 2023, as HPE has promised support for HP-UX 11.23
   (aka HP-UX 11i v2) only through the end of 2022; see Table 2 of
   <https://h20195.www2.hpe.com/V2/getpdf.aspx/4AA4-7673ENW.pdf>.  */
#ifdef __hpux
# undef UINT_LEAST8_MAX
# undef UINT_LEAST16_MAX
# define UINT_LEAST8_MAX 255
# define UINT_LEAST16_MAX 65535
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_uint8 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#if defined __GNUC__ && ! defined __ICC && 406 <= __GNUC__ * 100 + __GNUC_MINOR__
# if __GNUC__ * 100 + __GNUC_MINOR__ < 407
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")
# else
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# endif
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if !defined yyoverflow

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* !defined yyoverflow */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   220

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  45
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  47
/* YYNRULES -- Number of rules.  */
#define YYNRULES  100
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  249

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   299


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44
};

#if YYDEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,   203,   203,   203,   214,   213,   227,   237,   239,   241,
     243,   245,   247,   249,   251,   258,   257,   267,   270,   269,
     299,   298,   308,   311,   310,   336,   335,   365,   369,   364,
     380,   384,   383,   397,   396,   410,   409,   425,   429,   424,
     445,   488,   492,   505,   508,   527,   526,   533,   532,   547,
     555,   563,   571,   579,   587,   600,   606,   612,   618,   628,
     643,   663,   667,   679,   687,   695,   703,   711,   719,   732,
     738,   753,   772,   776,   788,   801,   809,   814,   820,   827,
     834,   841,   852,   858,   867,   874,   887,   889,   900,   906,
     916,   923,   934,   945,   947,   957,   968,   988,   990,   999,
    1010
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if YYDEBUG || 0
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "DEFINE_TOK",
  "DOMAIN_TOK", "REQUIREMENTS_TOK", "TYPES_TOK", "NUMBER_TOK",
  "EITHER_TOK", "CONSTANTS_TOK", "PREDICATES_TOK", "FUNCTIONS_TOK",
  "ACTION_TOK", "AXIOM_TOK", "VARS_TOK", "IMPLIES_TOK", "PRECONDITION_TOK",
  "PARAMETERS_TOK", "EFFECT_TOK", "AND_TOK", "NOT_TOK", "WHEN_TOK",
  "FORALL_TOK", "IMPLY_TOK", "OR_TOK", "EXISTS_TOK", "LE_TOK", "LEQ_TOK",
  "EQ_TOK", "GEQ_TOK", "GE_TOK", "MINUS_TOK", "AD_TOK", "MU_TOK", "DI_TOK",
  "ASSIGN_TOK", "SCALE_UP_TOK", "SCALE_DOWN_TOK", "INCREASE_TOK",
  "DECREASE_TOK", "NAME", "VARIABLE", "NUM", "OPEN_PAREN", "CLOSE_PAREN",
  "$accept", "file", "$@1", "domain_definition", "$@2", "domain_name",
  "optional_domain_defs", "predicates_def", "$@3", "predicates_list",
  "$@4", "functions_def", "$@5", "functions_list", "$@6", "$@7",
  "require_def", "$@8", "$@9", "require_key_star", "$@10", "types_def",
  "$@11", "constants_def", "$@12", "action_def", "$@13", "$@14",
  "axiom_def", "param_def", "action_def_body", "$@15", "$@16",
  "adl_goal_description", "adl_goal_description_star", "adl_effect",
  "adl_effect_star", "f_head", "f_exp", "literal_term",
  "atomic_formula_term", "term_star", "term", "name_plus", "predicate",
  "typed_list_name", "typed_list_variable", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#define YYPACT_NINF (-125)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-1)

#define yytable_value_is_error(Yyn) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int16 yypact[] =
{
    -125,    12,   -13,  -125,    30,  -125,    -7,    34,  -125,     2,
     -27,    11,   100,  -125,  -125,   -27,   -27,   -27,   -27,   -27,
     -27,   -27,  -125,  -125,  -125,  -125,    17,    18,  -125,    32,
    -125,  -125,  -125,  -125,  -125,  -125,  -125,    33,    36,    36,
      37,  -125,    39,  -125,    41,    48,  -125,    -6,    54,    55,
      59,    58,    59,    64,  -125,  -125,    59,    63,    74,    77,
    -125,  -125,  -125,    16,    76,  -125,    80,  -125,   101,    81,
    -125,    96,    74,    97,    36,    74,   102,  -125,  -125,    90,
     105,    13,   111,    63,  -125,  -125,    36,  -125,    99,    59,
      17,   148,    18,    59,   113,   111,   114,   115,    67,   129,
    -125,  -125,  -125,  -125,    59,  -125,  -125,  -125,  -125,   130,
      59,  -125,   125,  -125,  -125,  -125,   111,   111,   136,   111,
     111,   137,   -20,   -20,    87,   -20,   -20,     8,  -125,  -125,
      18,  -125,   138,    13,   114,   140,   111,   141,     8,   142,
     142,   142,   142,   142,    13,   111,   143,   144,   145,    59,
     111,   146,    59,  -125,    19,  -125,   -20,   -20,  -125,  -125,
     -20,   147,     8,   -20,   -20,   149,  -125,    13,  -125,   114,
     150,   -14,   145,   114,    59,   118,   -20,   -20,   -20,   -20,
     -20,  -125,  -125,  -125,  -125,  -125,   152,   153,  -125,   154,
     -20,   -20,   -20,   -20,     8,   155,   156,   157,  -125,  -125,
     158,   159,  -125,  -125,  -125,  -125,   160,   161,   162,   163,
     164,   165,   166,   111,  -125,   111,    27,   -20,   -20,   -20,
     167,  -125,  -125,  -125,  -125,  -125,  -125,   114,  -125,  -125,
    -125,  -125,  -125,   168,   169,  -125,   170,   171,   172,   173,
    -125,   174,  -125,  -125,  -125,  -125,  -125,  -125,  -125
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_int8 yydefact[] =
{
       2,     0,     0,     1,     0,     3,     0,     0,     4,     0,
       0,     0,     0,     7,     5,     0,     0,     0,     0,     0,
       0,     0,     6,    27,    33,    35,    17,    22,    37,     0,
      11,    12,     8,    10,     9,    13,    14,     0,    93,    93,
       0,    15,     0,    20,     0,     0,    28,    93,     0,     0,
      97,     0,    97,     0,    38,    92,    97,    30,     0,     0,
      96,    34,    36,    97,     0,    16,     0,    21,    41,     0,
      31,     0,    90,     0,    93,     0,     0,   100,    18,    23,
       0,    43,     0,    30,    29,    91,    93,    95,     0,    97,
      17,     0,    22,    97,     0,     0,     0,     0,     0,     0,
      54,    83,    32,    94,    97,    99,    19,    25,    24,     0,
      97,    45,     0,    47,    68,    39,    61,     0,     0,     0,
      61,     0,     0,     0,    86,     0,     0,    86,    40,    98,
      22,    42,     0,    43,    72,     0,     0,     0,    86,     0,
       0,     0,     0,     0,    43,    61,     0,     0,     0,    97,
       0,     0,    97,    75,     0,    76,     0,     0,    88,    89,
       0,     0,    86,     0,     0,     0,    26,    43,    46,    72,
       0,     0,     0,     0,    97,     0,     0,     0,     0,     0,
       0,    48,    62,    55,    57,    82,     0,     0,    56,     0,
       0,     0,     0,     0,    86,     0,     0,     0,    85,    87,
       0,     0,    84,    44,    73,    69,     0,     0,     0,     0,
       0,     0,     0,     0,    58,     0,     0,     0,     0,     0,
       0,    49,    50,    51,    52,    53,    71,     0,    63,    64,
      65,    66,    67,     0,     0,    77,     0,     0,     0,     0,
      74,     0,    60,    59,    79,    78,    80,    81,    70
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -125,  -125,  -125,  -125,  -125,  -125,   151,  -125,  -125,    91,
    -125,  -125,  -125,   -86,  -125,  -125,  -125,  -125,  -125,   103,
    -125,  -125,  -125,  -125,  -125,  -125,  -125,  -125,  -125,  -125,
    -124,  -125,  -125,   -82,  -113,   -93,    23,   -76,   -41,   -95,
    -107,  -122,  -125,   -54,   175,   -28,   -48
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_uint8 yydefgoto[] =
{
       0,     1,     2,     5,    10,     8,    14,    15,    51,    41,
      90,    16,    53,    43,    92,   130,    17,    37,    57,    71,
      83,    18,    38,    19,    39,    20,    44,    68,    21,    81,
      97,   133,   144,   145,   146,   169,   170,   155,   156,   100,
     101,   161,   162,    73,   127,    48,    64
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint8 yytable[] =
{
      99,   114,    58,   113,    66,   165,   108,   151,    69,   168,
     148,    49,     3,   111,   138,    77,    12,    13,    85,    60,
     181,    88,   153,   154,    75,    59,    55,    94,   172,    95,
       4,    96,   182,     6,    47,   147,     7,   150,     9,   114,
     199,   105,    11,   203,   166,   109,    87,    76,   158,   159,
     190,   191,   192,   193,   173,    22,   129,    63,   103,   194,
      40,    42,   132,   176,   177,   178,   179,   180,   187,   153,
     154,   235,   220,    46,   114,    45,    47,    50,   114,    52,
     206,    54,   157,   160,   163,   164,   116,   117,    55,   118,
     119,   120,   121,   122,   123,   124,   125,   126,    61,    62,
      63,   186,    65,    70,   189,    23,    24,    55,    67,    25,
      26,    27,    28,    29,    72,   195,   196,    74,    80,   197,
      78,    91,   200,   201,    79,    82,   207,   158,   159,   153,
     154,   233,   114,   234,   241,   208,   209,   210,   211,   212,
      84,    86,    89,   104,   134,   135,   136,   137,    93,   216,
     217,   218,   219,   138,    98,   107,   110,   112,   194,   115,
     139,   140,   141,   142,   143,    55,    30,    31,    32,    33,
      34,    35,    36,   128,   131,   236,   237,   238,   239,   149,
     152,   106,   167,   171,   174,   175,   102,   183,   184,   185,
     188,   198,   204,   202,   205,     0,   213,   214,   215,   221,
     222,   223,   224,   225,   226,   227,   228,   229,   230,   231,
     232,   240,   242,   243,   244,   245,   246,   247,   248,     0,
      56
};

static const yytype_int16 yycheck[] =
{
      82,    96,     8,    96,    52,   127,    92,   120,    56,   133,
     117,    39,     0,    95,    28,    63,    43,    44,    72,    47,
     144,    75,    42,    43,     8,    31,    40,    14,   135,    16,
      43,    18,   145,     3,    40,   117,    43,   119,     4,   134,
     162,    89,    40,   167,   130,    93,    74,    31,    40,    41,
      31,    32,    33,    34,   136,    44,   104,    41,    86,    40,
      43,    43,   110,   139,   140,   141,   142,   143,   150,    42,
      43,    44,   194,    40,   169,    43,    40,    40,   173,    40,
     173,    40,   123,   124,   125,   126,    19,    20,    40,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    44,    44,
      41,   149,    44,    40,   152,     5,     6,    40,    44,     9,
      10,    11,    12,    13,    40,   156,   157,    40,    17,   160,
      44,    31,   163,   164,    44,    44,   174,    40,    41,    42,
      43,   213,   227,   215,   227,   176,   177,   178,   179,   180,
      44,    44,    40,    44,    19,    20,    21,    22,    43,   190,
     191,   192,   193,    28,    43,     7,    43,    43,    40,    44,
      35,    36,    37,    38,    39,    40,    15,    16,    17,    18,
      19,    20,    21,    44,    44,   216,   217,   218,   219,    43,
      43,    90,    44,    43,    43,    43,    83,    44,    44,    44,
      44,    44,   169,    44,    44,    -1,    44,    44,    44,    44,
      44,    44,    44,    44,    44,    44,    44,    44,    44,    44,
      44,    44,    44,    44,    44,    44,    44,    44,    44,    -1,
      45
};

/* YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
   state STATE-NUM.  */
static const yytype_int8 yystos[] =
{
       0,    46,    47,     0,    43,    48,     3,    43,    50,     4,
      49,    40,    43,    44,    51,    52,    56,    61,    66,    68,
      70,    73,    44,     5,     6,     9,    10,    11,    12,    13,
      51,    51,    51,    51,    51,    51,    51,    62,    67,    69,
      43,    54,    43,    58,    71,    43,    40,    40,    90,    90,
      40,    53,    40,    57,    40,    40,    89,    63,     8,    31,
      90,    44,    44,    41,    91,    44,    91,    44,    72,    91,
      40,    64,    40,    88,    40,     8,    31,    91,    44,    44,
      17,    74,    44,    65,    44,    88,    44,    90,    88,    40,
      55,    31,    59,    43,    14,    16,    18,    75,    43,    78,
      84,    85,    64,    90,    44,    91,    54,     7,    58,    91,
      43,    78,    43,    80,    84,    44,    19,    20,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    89,    44,    91,
      60,    44,    91,    76,    19,    20,    21,    22,    28,    35,
      36,    37,    38,    39,    77,    78,    79,    78,    85,    43,
      78,    79,    43,    42,    43,    82,    83,    83,    40,    41,
      83,    86,    87,    83,    83,    86,    58,    44,    75,    80,
      81,    43,    85,    78,    43,    43,    82,    82,    82,    82,
      82,    75,    79,    44,    44,    44,    91,    78,    44,    91,
      31,    32,    33,    34,    40,    83,    83,    83,    44,    86,
      83,    83,    44,    75,    81,    44,    80,    91,    83,    83,
      83,    83,    83,    44,    44,    44,    83,    83,    83,    83,
      86,    44,    44,    44,    44,    44,    44,    44,    44,    44,
      44,    44,    44,    78,    78,    44,    83,    83,    83,    83,
      44,    80,    44,    44,    44,    44,    44,    44,    44
};

/* YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr1[] =
{
       0,    45,    47,    46,    49,    48,    50,    51,    51,    51,
      51,    51,    51,    51,    51,    53,    52,    54,    55,    54,
      57,    56,    58,    59,    58,    60,    58,    62,    63,    61,
      64,    65,    64,    67,    66,    69,    68,    71,    72,    70,
      73,    74,    74,    75,    75,    76,    75,    77,    75,    78,
      78,    78,    78,    78,    78,    78,    78,    78,    78,    78,
      78,    79,    79,    80,    80,    80,    80,    80,    80,    80,
      80,    80,    81,    81,    82,    83,    83,    83,    83,    83,
      83,    83,    84,    84,    85,    85,    86,    86,    87,    87,
      88,    88,    89,    90,    90,    90,    90,    91,    91,    91,
      91
};

/* YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     0,     2,     0,     5,     4,     1,     2,     2,
       2,     2,     2,     2,     2,     0,     5,     0,     0,     6,
       0,     5,     0,     0,     6,     0,     8,     0,     0,     7,
       0,     0,     3,     0,     5,     0,     5,     0,     0,     8,
       8,     0,     4,     0,     5,     0,     4,     0,     4,     5,
       5,     5,     5,     5,     1,     4,     4,     4,     5,     7,
       7,     0,     2,     5,     5,     5,     5,     5,     1,     4,
       7,     5,     0,     2,     4,     1,     1,     4,     5,     5,
       5,     5,     4,     1,     4,     4,     0,     2,     1,     1,
       1,     2,     1,     0,     5,     4,     2,     0,     5,     4,
       2
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYNOMEM         goto yyexhaustedlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use YYerror or YYUNDEF. */
#define YYERRCODE YYUNDEF


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)




# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  FILE *yyoutput = yyo;
  YY_USE (yyoutput);
  if (!yyvaluep)
    return;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  yy_symbol_value_print (yyo, yykind, yyvaluep);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp,
                 int yyrule)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)]);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif






/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep)
{
  YY_USE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* Lookahead token kind.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = YYEMPTY; /* Cause a token to be read.  */

  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    YYNOMEM;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        YYNOMEM;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          YYNOMEM;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */


  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = YYEOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == YYerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = YYUNDEF;
      yytoken = YYSYMBOL_YYerror;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 2: /* $@1: %empty  */
#line 203 "scan-ops_pddl.y"
{ 
  opserr( DOMDEF_EXPECTED, NULL ); 
}
#line 1537 "scan-ops_pddl.tab.c"
    break;

  case 4: /* $@2: %empty  */
#line 214 "scan-ops_pddl.y"
{ 
}
#line 1544 "scan-ops_pddl.tab.c"
    break;

  case 5: /* domain_definition: OPEN_PAREN DEFINE_TOK domain_name $@2 optional_domain_defs  */
#line 217 "scan-ops_pddl.y"
{
  if ( gcmd_line.display_info >= 1 ) {
    printf("\ndomain '%s' defined\n", gdomain_name);
  }
}
#line 1554 "scan-ops_pddl.tab.c"
    break;

  case 6: /* domain_name: OPEN_PAREN DOMAIN_TOK NAME CLOSE_PAREN  */
#line 228 "scan-ops_pddl.y"
{ 
  gdomain_name = new_Token( strlen((yyvsp[-1].string))+1 );
  strcpy( gdomain_name, (yyvsp[-1].string));
}
#line 1563 "scan-ops_pddl.tab.c"
    break;

  case 15: /* $@3: %empty  */
#line 258 "scan-ops_pddl.y"
{
}
#line 1570 "scan-ops_pddl.tab.c"
    break;

  case 16: /* predicates_def: OPEN_PAREN PREDICATES_TOK predicates_list $@3 CLOSE_PAREN  */
#line 261 "scan-ops_pddl.y"
{ 
}
#line 1577 "scan-ops_pddl.tab.c"
    break;

  case 17: /* predicates_list: %empty  */
#line 267 "scan-ops_pddl.y"
{}
#line 1583 "scan-ops_pddl.tab.c"
    break;

  case 18: /* $@4: %empty  */
#line 270 "scan-ops_pddl.y"
{

  TypedListList *tll;

  if ( gparse_predicates ) {
    tll = gparse_predicates;
    while ( tll->next ) {
      tll = tll->next;
    }
    tll->next = new_TypedListList();
    tll = tll->next;
  } else {
    tll = new_TypedListList();
    gparse_predicates = tll;
  }

  tll->predicate = new_Token( strlen( (yyvsp[-2].string) ) + 1);
  strcpy( tll->predicate, (yyvsp[-2].string) );

  tll->args = (yyvsp[-1].pTypedList);

}
#line 1610 "scan-ops_pddl.tab.c"
    break;

  case 20: /* $@5: %empty  */
#line 299 "scan-ops_pddl.y"
{
}
#line 1617 "scan-ops_pddl.tab.c"
    break;

  case 21: /* functions_def: OPEN_PAREN FUNCTIONS_TOK functions_list $@5 CLOSE_PAREN  */
#line 302 "scan-ops_pddl.y"
{ 
}
#line 1624 "scan-ops_pddl.tab.c"
    break;

  case 22: /* functions_list: %empty  */
#line 308 "scan-ops_pddl.y"
{}
#line 1630 "scan-ops_pddl.tab.c"
    break;

  case 23: /* $@6: %empty  */
#line 311 "scan-ops_pddl.y"
{

  TypedListList *tll;

  if ( gparse_functions ) {
    tll = gparse_functions;
    while ( tll->next ) {
      tll = tll->next;
    }
    tll->next = new_TypedListList();
    tll = tll->next;
  } else {
    tll = new_TypedListList();
    gparse_functions = tll;
  }

  tll->predicate = new_Token( strlen( (yyvsp[-2].string) ) + 1);
  strcpy( tll->predicate, (yyvsp[-2].string) );

  tll->args = (yyvsp[-1].pTypedList);

}
#line 1657 "scan-ops_pddl.tab.c"
    break;

  case 25: /* $@7: %empty  */
#line 336 "scan-ops_pddl.y"
{

  TypedListList *tll;

  if ( gparse_functions ) {
    tll = gparse_functions;
    while ( tll->next ) {
      tll = tll->next;
    }
    tll->next = new_TypedListList();
    tll = tll->next;
  } else {
    tll = new_TypedListList();
    gparse_functions = tll;
  }

  tll->predicate = new_Token( strlen( (yyvsp[-4].string) ) + 1);
  strcpy( tll->predicate, (yyvsp[-4].string) );

  tll->args = (yyvsp[-3].pTypedList);

}
#line 1684 "scan-ops_pddl.tab.c"
    break;

  case 27: /* $@8: %empty  */
#line 365 "scan-ops_pddl.y"
{ 
  opserr( REQUIREM_EXPECTED, NULL ); 
}
#line 1692 "scan-ops_pddl.tab.c"
    break;

  case 28: /* $@9: %empty  */
#line 369 "scan-ops_pddl.y"
{ 
  if ( !supported( (yyvsp[0].string) ) ) {
    opserr( NOT_SUPPORTED, (yyvsp[0].string) );
    yyerror();
  }
}
#line 1703 "scan-ops_pddl.tab.c"
    break;

  case 31: /* $@10: %empty  */
#line 384 "scan-ops_pddl.y"
{ 
  if ( !supported( (yyvsp[0].string) ) ) {
    opserr( NOT_SUPPORTED, (yyvsp[0].string) );
    yyerror();
  }
}
#line 1714 "scan-ops_pddl.tab.c"
    break;

  case 33: /* $@11: %empty  */
#line 397 "scan-ops_pddl.y"
{ 
  opserr( TYPEDEF_EXPECTED, NULL ); 
}
#line 1722 "scan-ops_pddl.tab.c"
    break;

  case 34: /* types_def: OPEN_PAREN TYPES_TOK $@11 typed_list_name CLOSE_PAREN  */
#line 401 "scan-ops_pddl.y"
{
  gparse_types = (yyvsp[-1].pTypedList);
}
#line 1730 "scan-ops_pddl.tab.c"
    break;

  case 35: /* $@12: %empty  */
#line 410 "scan-ops_pddl.y"
{ 
  opserr( CONSTLIST_EXPECTED, NULL ); 
}
#line 1738 "scan-ops_pddl.tab.c"
    break;

  case 36: /* constants_def: OPEN_PAREN CONSTANTS_TOK $@12 typed_list_name CLOSE_PAREN  */
#line 414 "scan-ops_pddl.y"
{
  gparse_constants = (yyvsp[-1].pTypedList);
}
#line 1746 "scan-ops_pddl.tab.c"
    break;

  case 37: /* $@13: %empty  */
#line 425 "scan-ops_pddl.y"
{ 
  opserr( ACTION, NULL ); 
}
#line 1754 "scan-ops_pddl.tab.c"
    break;

  case 38: /* $@14: %empty  */
#line 429 "scan-ops_pddl.y"
{ 
  scur_op = new_PlOperator( (yyvsp[0].string) );
  scur_op->axiom = FALSE;
}
#line 1763 "scan-ops_pddl.tab.c"
    break;

  case 39: /* action_def: OPEN_PAREN ACTION_TOK $@13 NAME $@14 param_def action_def_body CLOSE_PAREN  */
#line 434 "scan-ops_pddl.y"
{
  scur_op->next = gloaded_ops;
  gloaded_ops = scur_op; 
}
#line 1772 "scan-ops_pddl.tab.c"
    break;

  case 40: /* axiom_def: OPEN_PAREN AXIOM_TOK OPEN_PAREN predicate typed_list_variable CLOSE_PAREN adl_goal_description CLOSE_PAREN  */
#line 446 "scan-ops_pddl.y"
{ 
  PlNode *pln;
  TypedList *tyl;
  TokenList *tmp, *prev;

  pln = new_PlNode(ATOM);
  tmp = new_TokenList();
  tmp->item = new_Token( strlen( (yyvsp[-4].pstring) )+1 ); 
  strcpy( tmp->item, (yyvsp[-4].pstring) );
  pln->atom = tmp;

  scur_op = new_PlOperator( (yyvsp[-4].pstring) );
  scur_op->axiom = TRUE;
  scur_op->effects = pln;

  scur_op->preconds = (yyvsp[-1].pPlNode);

  scur_op->params = NULL;
  scur_op->parse_params = (yyvsp[-3].pTypedList);
  prev = pln->atom;
  for (tyl = scur_op->parse_params; tyl; tyl = tyl->next) {
    /* to be able to distinguish params from :VARS 
     */
    scur_op->number_of_real_params++;

    tmp = new_TokenList();
    tmp->item = new_Token( strlen( tyl->name )+1 ); 
    strcpy( tmp->item, tyl->name );
    prev->next = tmp;
    prev = tmp;
    
  }
  
  scur_op->next = gloaded_ops;
  gloaded_ops = scur_op; 
}
#line 1813 "scan-ops_pddl.tab.c"
    break;

  case 41: /* param_def: %empty  */
#line 488 "scan-ops_pddl.y"
{ 
  scur_op->params = NULL; 
}
#line 1821 "scan-ops_pddl.tab.c"
    break;

  case 42: /* param_def: PARAMETERS_TOK OPEN_PAREN typed_list_variable CLOSE_PAREN  */
#line 493 "scan-ops_pddl.y"
{
  TypedList *tl;
  scur_op->parse_params = (yyvsp[-1].pTypedList);
  for (tl = scur_op->parse_params; tl; tl = tl->next) {
    /* to be able to distinguish params from :VARS 
     */
    scur_op->number_of_real_params++;
  }
}
#line 1835 "scan-ops_pddl.tab.c"
    break;

  case 44: /* action_def_body: VARS_TOK OPEN_PAREN typed_list_variable CLOSE_PAREN action_def_body  */
#line 509 "scan-ops_pddl.y"
{
  TypedList *tl = NULL;

  /* add vars as parameters 
   */
  if ( scur_op->parse_params ) {
    for( tl = scur_op->parse_params; tl->next; tl = tl->next ) {
      /* empty, get to the end of list 
       */
    }
    tl->next = (yyvsp[-2].pTypedList);
    tl = tl->next;
  } else {
    scur_op->parse_params = (yyvsp[-2].pTypedList);
  }
}
#line 1856 "scan-ops_pddl.tab.c"
    break;

  case 45: /* $@15: %empty  */
#line 527 "scan-ops_pddl.y"
{ 
  scur_op->preconds = (yyvsp[0].pPlNode); 
}
#line 1864 "scan-ops_pddl.tab.c"
    break;

  case 47: /* $@16: %empty  */
#line 533 "scan-ops_pddl.y"
{ 
  scur_op->effects = (yyvsp[0].pPlNode); 
}
#line 1872 "scan-ops_pddl.tab.c"
    break;

  case 49: /* adl_goal_description: OPEN_PAREN LE_TOK f_exp f_exp CLOSE_PAREN  */
#line 548 "scan-ops_pddl.y"
{
  (yyval.pPlNode) = new_PlNode(COMP);
  (yyval.pPlNode)->comp = LE;
  (yyval.pPlNode)->lh = (yyvsp[-2].pParseExpNode);
  (yyval.pPlNode)->rh = (yyvsp[-1].pParseExpNode);
}
#line 1883 "scan-ops_pddl.tab.c"
    break;

  case 50: /* adl_goal_description: OPEN_PAREN LEQ_TOK f_exp f_exp CLOSE_PAREN  */
#line 556 "scan-ops_pddl.y"
{
  (yyval.pPlNode) = new_PlNode(COMP);
  (yyval.pPlNode)->comp = LEQ;
  (yyval.pPlNode)->lh = (yyvsp[-2].pParseExpNode);
  (yyval.pPlNode)->rh = (yyvsp[-1].pParseExpNode);
}
#line 1894 "scan-ops_pddl.tab.c"
    break;

  case 51: /* adl_goal_description: OPEN_PAREN EQ_TOK f_exp f_exp CLOSE_PAREN  */
#line 564 "scan-ops_pddl.y"
{
  (yyval.pPlNode) = new_PlNode(COMP);
  (yyval.pPlNode)->comp = EQ;
  (yyval.pPlNode)->lh = (yyvsp[-2].pParseExpNode);
  (yyval.pPlNode)->rh = (yyvsp[-1].pParseExpNode);
}
#line 1905 "scan-ops_pddl.tab.c"
    break;

  case 52: /* adl_goal_description: OPEN_PAREN GEQ_TOK f_exp f_exp CLOSE_PAREN  */
#line 572 "scan-ops_pddl.y"
{
  (yyval.pPlNode) = new_PlNode(COMP);
  (yyval.pPlNode)->comp = GEQ;
  (yyval.pPlNode)->lh = (yyvsp[-2].pParseExpNode);
  (yyval.pPlNode)->rh = (yyvsp[-1].pParseExpNode);
}
#line 1916 "scan-ops_pddl.tab.c"
    break;

  case 53: /* adl_goal_description: OPEN_PAREN GE_TOK f_exp f_exp CLOSE_PAREN  */
#line 580 "scan-ops_pddl.y"
{
  (yyval.pPlNode) = new_PlNode(COMP);
  (yyval.pPlNode)->comp = GE;
  (yyval.pPlNode)->lh = (yyvsp[-2].pParseExpNode);
  (yyval.pPlNode)->rh = (yyvsp[-1].pParseExpNode);
}
#line 1927 "scan-ops_pddl.tab.c"
    break;

  case 54: /* adl_goal_description: literal_term  */
#line 588 "scan-ops_pddl.y"
{ 
  if ( sis_negated ) {
    (yyval.pPlNode) = new_PlNode(NOT);
    (yyval.pPlNode)->sons = new_PlNode(ATOM);
    (yyval.pPlNode)->sons->atom = (yyvsp[0].pTokenList);
    sis_negated = FALSE;
  } else {
    (yyval.pPlNode) = new_PlNode(ATOM);
    (yyval.pPlNode)->atom = (yyvsp[0].pTokenList);
  }
}
#line 1943 "scan-ops_pddl.tab.c"
    break;

  case 55: /* adl_goal_description: OPEN_PAREN AND_TOK adl_goal_description_star CLOSE_PAREN  */
#line 601 "scan-ops_pddl.y"
{ 
  (yyval.pPlNode) = new_PlNode(AND);
  (yyval.pPlNode)->sons = (yyvsp[-1].pPlNode);
}
#line 1952 "scan-ops_pddl.tab.c"
    break;

  case 56: /* adl_goal_description: OPEN_PAREN OR_TOK adl_goal_description_star CLOSE_PAREN  */
#line 607 "scan-ops_pddl.y"
{ 
  (yyval.pPlNode) = new_PlNode(OR);
  (yyval.pPlNode)->sons = (yyvsp[-1].pPlNode);
}
#line 1961 "scan-ops_pddl.tab.c"
    break;

  case 57: /* adl_goal_description: OPEN_PAREN NOT_TOK adl_goal_description CLOSE_PAREN  */
#line 613 "scan-ops_pddl.y"
{ 
  (yyval.pPlNode) = new_PlNode(NOT);
  (yyval.pPlNode)->sons = (yyvsp[-1].pPlNode);
}
#line 1970 "scan-ops_pddl.tab.c"
    break;

  case 58: /* adl_goal_description: OPEN_PAREN IMPLY_TOK adl_goal_description adl_goal_description CLOSE_PAREN  */
#line 619 "scan-ops_pddl.y"
{ 
  PlNode *np = new_PlNode(NOT);
  np->sons = (yyvsp[-2].pPlNode);
  np->next = (yyvsp[-1].pPlNode);

  (yyval.pPlNode) = new_PlNode(OR);
  (yyval.pPlNode)->sons = np;
}
#line 1983 "scan-ops_pddl.tab.c"
    break;

  case 59: /* adl_goal_description: OPEN_PAREN EXISTS_TOK OPEN_PAREN typed_list_variable CLOSE_PAREN adl_goal_description CLOSE_PAREN  */
#line 631 "scan-ops_pddl.y"
{ 

  PlNode *pln;

  pln = new_PlNode(EX);
  pln->parse_vars = (yyvsp[-3].pTypedList);

  (yyval.pPlNode) = pln;
  pln->sons = (yyvsp[-1].pPlNode);

}
#line 1999 "scan-ops_pddl.tab.c"
    break;

  case 60: /* adl_goal_description: OPEN_PAREN FORALL_TOK OPEN_PAREN typed_list_variable CLOSE_PAREN adl_goal_description CLOSE_PAREN  */
#line 646 "scan-ops_pddl.y"
{ 

  PlNode *pln;

  pln = new_PlNode(ALL);
  pln->parse_vars = (yyvsp[-3].pTypedList);

  (yyval.pPlNode) = pln;
  pln->sons = (yyvsp[-1].pPlNode);

}
#line 2015 "scan-ops_pddl.tab.c"
    break;

  case 61: /* adl_goal_description_star: %empty  */
#line 663 "scan-ops_pddl.y"
{
  (yyval.pPlNode) = NULL;
}
#line 2023 "scan-ops_pddl.tab.c"
    break;

  case 62: /* adl_goal_description_star: adl_goal_description adl_goal_description_star  */
#line 668 "scan-ops_pddl.y"
{
  (yyvsp[-1].pPlNode)->next = (yyvsp[0].pPlNode);
  (yyval.pPlNode) = (yyvsp[-1].pPlNode);
}
#line 2032 "scan-ops_pddl.tab.c"
    break;

  case 63: /* adl_effect: OPEN_PAREN ASSIGN_TOK f_head f_exp CLOSE_PAREN  */
#line 680 "scan-ops_pddl.y"
{
  (yyval.pPlNode) = new_PlNode( NEF );
  (yyval.pPlNode)->neft = ASSIGN;
  (yyval.pPlNode)->lh = (yyvsp[-2].pParseExpNode);
  (yyval.pPlNode)->rh = (yyvsp[-1].pParseExpNode);
}
#line 2043 "scan-ops_pddl.tab.c"
    break;

  case 64: /* adl_effect: OPEN_PAREN SCALE_UP_TOK f_head f_exp CLOSE_PAREN  */
#line 688 "scan-ops_pddl.y"
{
  (yyval.pPlNode) = new_PlNode( NEF );
  (yyval.pPlNode)->neft = SCALE_UP;
  (yyval.pPlNode)->lh = (yyvsp[-2].pParseExpNode);
  (yyval.pPlNode)->rh = (yyvsp[-1].pParseExpNode);
}
#line 2054 "scan-ops_pddl.tab.c"
    break;

  case 65: /* adl_effect: OPEN_PAREN SCALE_DOWN_TOK f_head f_exp CLOSE_PAREN  */
#line 696 "scan-ops_pddl.y"
{
  (yyval.pPlNode) = new_PlNode( NEF );
  (yyval.pPlNode)->neft = SCALE_DOWN;
  (yyval.pPlNode)->lh = (yyvsp[-2].pParseExpNode);
  (yyval.pPlNode)->rh = (yyvsp[-1].pParseExpNode);
}
#line 2065 "scan-ops_pddl.tab.c"
    break;

  case 66: /* adl_effect: OPEN_PAREN INCREASE_TOK f_head f_exp CLOSE_PAREN  */
#line 704 "scan-ops_pddl.y"
{
  (yyval.pPlNode) = new_PlNode( NEF );
  (yyval.pPlNode)->neft = INCREASE;
  (yyval.pPlNode)->lh = (yyvsp[-2].pParseExpNode);
  (yyval.pPlNode)->rh = (yyvsp[-1].pParseExpNode);
}
#line 2076 "scan-ops_pddl.tab.c"
    break;

  case 67: /* adl_effect: OPEN_PAREN DECREASE_TOK f_head f_exp CLOSE_PAREN  */
#line 712 "scan-ops_pddl.y"
{
  (yyval.pPlNode) = new_PlNode( NEF );
  (yyval.pPlNode)->neft = DECREASE;
  (yyval.pPlNode)->lh = (yyvsp[-2].pParseExpNode);
  (yyval.pPlNode)->rh = (yyvsp[-1].pParseExpNode);
}
#line 2087 "scan-ops_pddl.tab.c"
    break;

  case 68: /* adl_effect: literal_term  */
#line 720 "scan-ops_pddl.y"
{ 
  if ( sis_negated ) {
    (yyval.pPlNode) = new_PlNode(NOT);
    (yyval.pPlNode)->sons = new_PlNode(ATOM);
    (yyval.pPlNode)->sons->atom = (yyvsp[0].pTokenList);
    sis_negated = FALSE;
  } else {
    (yyval.pPlNode) = new_PlNode(ATOM);
    (yyval.pPlNode)->atom = (yyvsp[0].pTokenList);
  }
}
#line 2103 "scan-ops_pddl.tab.c"
    break;

  case 69: /* adl_effect: OPEN_PAREN AND_TOK adl_effect_star CLOSE_PAREN  */
#line 733 "scan-ops_pddl.y"
{ 
  (yyval.pPlNode) = new_PlNode(AND);
  (yyval.pPlNode)->sons = (yyvsp[-1].pPlNode);
}
#line 2112 "scan-ops_pddl.tab.c"
    break;

  case 70: /* adl_effect: OPEN_PAREN FORALL_TOK OPEN_PAREN typed_list_variable CLOSE_PAREN adl_effect CLOSE_PAREN  */
#line 741 "scan-ops_pddl.y"
{ 

  PlNode *pln;

  pln = new_PlNode(ALL);
  pln->parse_vars = (yyvsp[-3].pTypedList);

  (yyval.pPlNode) = pln;
  pln->sons = (yyvsp[-1].pPlNode);

}
#line 2128 "scan-ops_pddl.tab.c"
    break;

  case 71: /* adl_effect: OPEN_PAREN WHEN_TOK adl_goal_description adl_effect CLOSE_PAREN  */
#line 754 "scan-ops_pddl.y"
{
  /* This will be conditional effects in FF representation, but here
   * a formula like (WHEN p q) will be saved as:
   *  [WHEN]
   *  [sons]
   *   /  \
   * [p]  [q]
   * That means, the first son is p, and the second one is q. 
   */
  (yyval.pPlNode) = new_PlNode(WHEN);
  (yyvsp[-2].pPlNode)->next = (yyvsp[-1].pPlNode);
  (yyval.pPlNode)->sons = (yyvsp[-2].pPlNode);
}
#line 2146 "scan-ops_pddl.tab.c"
    break;

  case 72: /* adl_effect_star: %empty  */
#line 772 "scan-ops_pddl.y"
{ 
  (yyval.pPlNode) = NULL; 
}
#line 2154 "scan-ops_pddl.tab.c"
    break;

  case 73: /* adl_effect_star: adl_effect adl_effect_star  */
#line 777 "scan-ops_pddl.y"
{
  (yyvsp[-1].pPlNode)->next = (yyvsp[0].pPlNode);
  (yyval.pPlNode) = (yyvsp[-1].pPlNode);
}
#line 2163 "scan-ops_pddl.tab.c"
    break;

  case 74: /* f_head: OPEN_PAREN NAME term_star CLOSE_PAREN  */
#line 789 "scan-ops_pddl.y"
{
  (yyval.pParseExpNode) = new_ParseExpNode( FHEAD );
  (yyval.pParseExpNode)->atom = new_TokenList();
  (yyval.pParseExpNode)->atom->item = new_Token( strlen((yyvsp[-2].string))+1 );
  strcpy( (yyval.pParseExpNode)->atom->item, (yyvsp[-2].string) );
  (yyval.pParseExpNode)->atom->next = (yyvsp[-1].pTokenList);
}
#line 2175 "scan-ops_pddl.tab.c"
    break;

  case 75: /* f_exp: NUM  */
#line 802 "scan-ops_pddl.y"
{ 
  (yyval.pParseExpNode) = new_ParseExpNode( NUMBER );
  (yyval.pParseExpNode)->atom = new_TokenList();
  (yyval.pParseExpNode)->atom->item = new_Token( strlen((yyvsp[0].string))+1 );
  strcpy( (yyval.pParseExpNode)->atom->item, (yyvsp[0].string) );
}
#line 2186 "scan-ops_pddl.tab.c"
    break;

  case 76: /* f_exp: f_head  */
#line 810 "scan-ops_pddl.y"
{
  (yyval.pParseExpNode) = (yyvsp[0].pParseExpNode);
}
#line 2194 "scan-ops_pddl.tab.c"
    break;

  case 77: /* f_exp: OPEN_PAREN MINUS_TOK f_exp CLOSE_PAREN  */
#line 815 "scan-ops_pddl.y"
{
  (yyval.pParseExpNode) = new_ParseExpNode( MINUS );
  (yyval.pParseExpNode)->leftson = (yyvsp[-1].pParseExpNode);
}
#line 2203 "scan-ops_pddl.tab.c"
    break;

  case 78: /* f_exp: OPEN_PAREN AD_TOK f_exp f_exp CLOSE_PAREN  */
#line 821 "scan-ops_pddl.y"
{
  (yyval.pParseExpNode) = new_ParseExpNode( AD );
  (yyval.pParseExpNode)->leftson = (yyvsp[-2].pParseExpNode);
  (yyval.pParseExpNode)->rightson = (yyvsp[-1].pParseExpNode);
}
#line 2213 "scan-ops_pddl.tab.c"
    break;

  case 79: /* f_exp: OPEN_PAREN MINUS_TOK f_exp f_exp CLOSE_PAREN  */
#line 828 "scan-ops_pddl.y"
{
  (yyval.pParseExpNode) = new_ParseExpNode( SU );
  (yyval.pParseExpNode)->leftson = (yyvsp[-2].pParseExpNode);
  (yyval.pParseExpNode)->rightson = (yyvsp[-1].pParseExpNode);
}
#line 2223 "scan-ops_pddl.tab.c"
    break;

  case 80: /* f_exp: OPEN_PAREN MU_TOK f_exp f_exp CLOSE_PAREN  */
#line 835 "scan-ops_pddl.y"
{
  (yyval.pParseExpNode) = new_ParseExpNode( MU );
  (yyval.pParseExpNode)->leftson = (yyvsp[-2].pParseExpNode);
  (yyval.pParseExpNode)->rightson = (yyvsp[-1].pParseExpNode);
}
#line 2233 "scan-ops_pddl.tab.c"
    break;

  case 81: /* f_exp: OPEN_PAREN DI_TOK f_exp f_exp CLOSE_PAREN  */
#line 842 "scan-ops_pddl.y"
{
  (yyval.pParseExpNode) = new_ParseExpNode( DI );
  (yyval.pParseExpNode)->leftson = (yyvsp[-2].pParseExpNode);
  (yyval.pParseExpNode)->rightson = (yyvsp[-1].pParseExpNode);
}
#line 2243 "scan-ops_pddl.tab.c"
    break;

  case 82: /* literal_term: OPEN_PAREN NOT_TOK atomic_formula_term CLOSE_PAREN  */
#line 853 "scan-ops_pddl.y"
{ 
  (yyval.pTokenList) = (yyvsp[-1].pTokenList);
  sis_negated = TRUE;
}
#line 2252 "scan-ops_pddl.tab.c"
    break;

  case 83: /* literal_term: atomic_formula_term  */
#line 859 "scan-ops_pddl.y"
{
  (yyval.pTokenList) = (yyvsp[0].pTokenList);
}
#line 2260 "scan-ops_pddl.tab.c"
    break;

  case 84: /* atomic_formula_term: OPEN_PAREN predicate term_star CLOSE_PAREN  */
#line 868 "scan-ops_pddl.y"
{ 
  (yyval.pTokenList) = new_TokenList();
  (yyval.pTokenList)->item = (yyvsp[-2].pstring);
  (yyval.pTokenList)->next = (yyvsp[-1].pTokenList);
}
#line 2270 "scan-ops_pddl.tab.c"
    break;

  case 85: /* atomic_formula_term: OPEN_PAREN EQ_TOK term_star CLOSE_PAREN  */
#line 875 "scan-ops_pddl.y"
{
  (yyval.pTokenList) = new_TokenList();
  (yyval.pTokenList)->item = new_Token( 5 );
  (yyval.pTokenList)->item = "=";
  (yyval.pTokenList)->next = (yyvsp[-1].pTokenList);
}
#line 2281 "scan-ops_pddl.tab.c"
    break;

  case 86: /* term_star: %empty  */
#line 887 "scan-ops_pddl.y"
{ (yyval.pTokenList) = NULL; }
#line 2287 "scan-ops_pddl.tab.c"
    break;

  case 87: /* term_star: term term_star  */
#line 890 "scan-ops_pddl.y"
{
  (yyval.pTokenList) = new_TokenList();
  (yyval.pTokenList)->item = (yyvsp[-1].pstring);
  (yyval.pTokenList)->next = (yyvsp[0].pTokenList);
}
#line 2297 "scan-ops_pddl.tab.c"
    break;

  case 88: /* term: NAME  */
#line 901 "scan-ops_pddl.y"
{ 
  (yyval.pstring) = new_Token( strlen((yyvsp[0].string))+1 );
  strcpy( (yyval.pstring), (yyvsp[0].string) );
}
#line 2306 "scan-ops_pddl.tab.c"
    break;

  case 89: /* term: VARIABLE  */
#line 907 "scan-ops_pddl.y"
{ 
  (yyval.pstring) = new_Token( strlen((yyvsp[0].string))+1 );
  strcpy( (yyval.pstring), (yyvsp[0].string) );
}
#line 2315 "scan-ops_pddl.tab.c"
    break;

  case 90: /* name_plus: NAME  */
#line 917 "scan-ops_pddl.y"
{
  (yyval.pTokenList) = new_TokenList();
  (yyval.pTokenList)->item = new_Token( strlen((yyvsp[0].string))+1 );
  strcpy( (yyval.pTokenList)->item, (yyvsp[0].string) );
}
#line 2325 "scan-ops_pddl.tab.c"
    break;

  case 91: /* name_plus: NAME name_plus  */
#line 924 "scan-ops_pddl.y"
{
  (yyval.pTokenList) = new_TokenList();
  (yyval.pTokenList)->item = new_Token( strlen((yyvsp[-1].string))+1 );
  strcpy( (yyval.pTokenList)->item, (yyvsp[-1].string) );
  (yyval.pTokenList)->next = (yyvsp[0].pTokenList);
}
#line 2336 "scan-ops_pddl.tab.c"
    break;

  case 92: /* predicate: NAME  */
#line 935 "scan-ops_pddl.y"
{ 
  (yyval.pstring) = new_Token( strlen((yyvsp[0].string))+1 );
  strcpy( (yyval.pstring), (yyvsp[0].string) );
}
#line 2345 "scan-ops_pddl.tab.c"
    break;

  case 93: /* typed_list_name: %empty  */
#line 945 "scan-ops_pddl.y"
{ (yyval.pTypedList) = NULL; }
#line 2351 "scan-ops_pddl.tab.c"
    break;

  case 94: /* typed_list_name: NAME EITHER_TOK name_plus CLOSE_PAREN typed_list_name  */
#line 948 "scan-ops_pddl.y"
{ 

  (yyval.pTypedList) = new_TypedList();
  (yyval.pTypedList)->name = new_Token( strlen((yyvsp[-4].string))+1 );
  strcpy( (yyval.pTypedList)->name, (yyvsp[-4].string) );
  (yyval.pTypedList)->type = (yyvsp[-2].pTokenList);
  (yyval.pTypedList)->next = (yyvsp[0].pTypedList);
}
#line 2364 "scan-ops_pddl.tab.c"
    break;

  case 95: /* typed_list_name: NAME MINUS_TOK NAME typed_list_name  */
#line 958 "scan-ops_pddl.y"
{
  (yyval.pTypedList) = new_TypedList();
  (yyval.pTypedList)->name = new_Token( strlen((yyvsp[-3].string))+1 );
  strcpy( (yyval.pTypedList)->name, (yyvsp[-3].string) );
  (yyval.pTypedList)->type = new_TokenList();
  (yyval.pTypedList)->type->item = new_Token( strlen((yyvsp[-1].string))+1 );
  strcpy( (yyval.pTypedList)->type->item, (yyvsp[-1].string) );
  (yyval.pTypedList)->next = (yyvsp[0].pTypedList);
}
#line 2378 "scan-ops_pddl.tab.c"
    break;

  case 96: /* typed_list_name: NAME typed_list_name  */
#line 969 "scan-ops_pddl.y"
{
  (yyval.pTypedList) = new_TypedList();
  (yyval.pTypedList)->name = new_Token( strlen((yyvsp[-1].string))+1 );
  strcpy( (yyval.pTypedList)->name, (yyvsp[-1].string) );
  if ( (yyvsp[0].pTypedList) ) {/* another element (already typed) is following */
    (yyval.pTypedList)->type = copy_TokenList( (yyvsp[0].pTypedList)->type );
  } else {/* no further element - it must be an untyped list */
    (yyval.pTypedList)->type = new_TokenList();
    (yyval.pTypedList)->type->item = new_Token( strlen(STANDARD_TYPE)+1 );
    strcpy( (yyval.pTypedList)->type->item, STANDARD_TYPE );
  }
  (yyval.pTypedList)->next = (yyvsp[0].pTypedList);
}
#line 2396 "scan-ops_pddl.tab.c"
    break;

  case 97: /* typed_list_variable: %empty  */
#line 988 "scan-ops_pddl.y"
{ (yyval.pTypedList) = NULL; }
#line 2402 "scan-ops_pddl.tab.c"
    break;

  case 98: /* typed_list_variable: VARIABLE EITHER_TOK name_plus CLOSE_PAREN typed_list_variable  */
#line 991 "scan-ops_pddl.y"
{ 
  (yyval.pTypedList) = new_TypedList();
  (yyval.pTypedList)->name = new_Token( strlen((yyvsp[-4].string))+1 );
  strcpy( (yyval.pTypedList)->name, (yyvsp[-4].string) );
  (yyval.pTypedList)->type = (yyvsp[-2].pTokenList);
  (yyval.pTypedList)->next = (yyvsp[0].pTypedList);
}
#line 2414 "scan-ops_pddl.tab.c"
    break;

  case 99: /* typed_list_variable: VARIABLE MINUS_TOK NAME typed_list_variable  */
#line 1000 "scan-ops_pddl.y"
{
  (yyval.pTypedList) = new_TypedList();
  (yyval.pTypedList)->name = new_Token( strlen((yyvsp[-3].string))+1 );
  strcpy( (yyval.pTypedList)->name, (yyvsp[-3].string) );
  (yyval.pTypedList)->type = new_TokenList();
  (yyval.pTypedList)->type->item = new_Token( strlen((yyvsp[-1].string))+1 );
  strcpy( (yyval.pTypedList)->type->item, (yyvsp[-1].string) );
  (yyval.pTypedList)->next = (yyvsp[0].pTypedList);
}
#line 2428 "scan-ops_pddl.tab.c"
    break;

  case 100: /* typed_list_variable: VARIABLE typed_list_variable  */
#line 1011 "scan-ops_pddl.y"
{
  (yyval.pTypedList) = new_TypedList();
  (yyval.pTypedList)->name = new_Token( strlen((yyvsp[-1].string))+1 );
  strcpy( (yyval.pTypedList)->name, (yyvsp[-1].string) );
  if ( (yyvsp[0].pTypedList) ) {/* another element (already typed) is following */
    (yyval.pTypedList)->type = copy_TokenList( (yyvsp[0].pTypedList)->type );
  } else {/* no further element - it must be an untyped list */
    (yyval.pTypedList)->type = new_TokenList();
    (yyval.pTypedList)->type->item = new_Token( strlen(STANDARD_TYPE)+1 );
    strcpy( (yyval.pTypedList)->type->item, STANDARD_TYPE );
  }
  (yyval.pTypedList)->next = (yyvsp[0].pTypedList);
}
#line 2446 "scan-ops_pddl.tab.c"
    break;


#line 2450 "scan-ops_pddl.tab.c"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      yyerror (YY_("syntax error"));
    }

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;
  ++yynerrs;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturnlab;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturnlab;


/*-----------------------------------------------------------.
| yyexhaustedlab -- YYNOMEM (memory exhaustion) comes here.  |
`-----------------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturnlab;


/*----------------------------------------------------------.
| yyreturnlab -- parsing is finished, clean up and return.  |
`----------------------------------------------------------*/
yyreturnlab:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif

  return yyresult;
}

#line 1028 "scan-ops_pddl.y"

#include "lex.ops_pddl.c"


/**********************************************************************
 * Functions
 **********************************************************************/

/* 
 * call	bison -pops -bscan-ops scan-ops.y
 */

  


int yyerror( char *msg )

{

  fflush(stdout);
  fprintf(stderr, "\n%s: syntax error in line %d, '%s':\n", 
	  gact_filename, lineno, yytext);

  if ( NULL != sact_err_par ) {
    fprintf(stderr, "%s %s\n", serrmsg[sact_err], sact_err_par);
  } else {
    fprintf(stderr, "%s\n", serrmsg[sact_err]);
  }

  exit( 1 );

}



void load_ops_file( char *filename )

{

  FILE * fp;/* pointer to input files */
  char tmp[MAX_LENGTH] = "";

  /* open operator file 
   */
  if( ( fp = fopen( filename, "r" ) ) == NULL ) {
    sprintf(tmp, "\nff: can't find operator file: %s\n\n", filename );
    perror(tmp);
    exit( 1 );
  }

  gact_filename = filename;
  lineno = 1; 
  yyin = fp;

  yyparse();

  fclose( fp );/* and close file again */

}
