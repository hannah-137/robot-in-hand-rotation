"""Check which mjlab and rsl_rl APIs used by this repo changed between two versions.

Usage:
  python scripts/check_mjlab_api.py \
    --pkg mjlab <old mjlab source> <new mjlab source> \
    --pkg rsl_rl <old rsl_rl source> <new rsl_rl source>

A source folder is an extracted release archive, for example:
  https://github.com/mujocolab/mjlab/archive/refs/tags/v1.6.0.tar.gz
  https://github.com/leggedrobotics/rsl_rl/archive/refs/tags/v5.4.2.tar.gz

The script only reads files. It does not import the packages. It prints Markdown.

Checks on the repo code:
  1. Dependency changes in the pyproject.toml of each package.
  2. Imported names and attribute paths that exist in the old version but not in the new one.
  3. Keyword arguments that the new version does not accept. This includes the keys
     of a "params" dict next to "func" (manager term configs).
  4. Methods that repo classes override when the base method changed its parameters.
  5. Abstract methods that are new in a base class and missing in the repo class.
  6. "<x>.data.<name>" attributes that exist in an old "*Data" class but in no new one.

A static check cannot find every problem. Run the code after you fix these items.
"""

from __future__ import annotations

import argparse
import ast
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "__pycache__", "logs", "wandb", "outputs"}
FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass
class Sig:
  """Parameter names of a callable. var_kw is True if it takes **kwargs."""

  names: list[str]
  var_kw: bool


@dataclass(frozen=True)
class Finding:
  check: str
  file: str
  line: int
  api: str
  detail: str


def attr_chain(expr: ast.AST | None) -> list[str] | None:
  """Turn "a.b.c" into ["a", "b", "c"]. Return None for other expressions."""
  parts: list[str] = []
  while isinstance(expr, ast.Attribute):
    parts.append(expr.attr)
    expr = expr.value
  if isinstance(expr, ast.Name):
    parts.append(expr.id)
    return parts[::-1]
  return None


def target_names(target: ast.AST) -> list[str]:
  if isinstance(target, ast.Name):
    return [target.id]
  if isinstance(target, (ast.Tuple, ast.List)):
    return [name for elt in target.elts for name in target_names(elt)]
  return []


def func_sig(node: ast.FunctionDef | ast.AsyncFunctionDef, drop_first: bool) -> Sig:
  args = node.args
  positional = [a.arg for a in args.posonlyargs + args.args]
  if drop_first:
    positional = positional[1:]
  return Sig(positional + [a.arg for a in args.kwonlyargs], args.kwarg is not None)


def decorator_names(node: ast.AST) -> list[str]:
  names = []
  for dec in getattr(node, "decorator_list", []):
    chain = attr_chain(dec.func if isinstance(dec, ast.Call) else dec)
    if chain:
      names.append(chain[-1])
  return names


def is_classvar(node: ast.AnnAssign) -> bool:
  ann = node.annotation
  chain = attr_chain(ann.value if isinstance(ann, ast.Subscript) else ann)
  return bool(chain) and chain[-1] == "ClassVar"


def is_type_checking(test: ast.expr) -> bool:
  chain = attr_chain(test)
  return bool(chain) and chain[-1] == "TYPE_CHECKING"


class PackageIndex:
  """A package read from its source folder. It resolves names without importing."""

  def __init__(self, name: str, source: Path):
    self.name = name
    self.source = source
    self.pkg_dir = self._find_pkg_dir(name, source)
    self._trees: dict[str, ast.Module | None] = {}
    self._symbols: dict[str, dict[str, tuple] | None] = {}
    self._members: dict[tuple, dict[str, tuple]] = {}
    self._data_members: set[str] | None = None

  @staticmethod
  def _find_pkg_dir(name: str, source: Path) -> Path:
    for candidate in (source / "src" / name, source / name):
      if (candidate / "__init__.py").exists():
        return candidate
    raise SystemExit(f"error: package '{name}' not found in {source}")

  def version(self) -> str:
    path = self.source / "pyproject.toml"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    return match.group(1) if match else "?"

  def owns(self, module: str) -> bool:
    return module == self.name or module.startswith(self.name + ".")

  def module_file(self, module: str) -> Path | None:
    if not self.owns(module):
      return None
    rel = module.split(".")[1:]
    base = self.pkg_dir.joinpath(*rel)
    if (base / "__init__.py").exists():
      return base / "__init__.py"
    if rel and base.with_suffix(".py").exists():
      return base.with_suffix(".py")
    return None

  def module_ref(self, module: str) -> tuple | None:
    if not self.owns(module):
      return ("unknown", module)
    return ("module", module) if self.module_file(module) else None

  def tree(self, module: str) -> ast.Module | None:
    if module not in self._trees:
      path = self.module_file(module)
      self._trees[module] = (
        ast.parse(path.read_text(encoding="utf-8"), str(path)) if path else None
      )
    return self._trees[module]

  def _absolute(self, module: str, node: ast.ImportFrom) -> str:
    if node.level == 0:
      return node.module or ""
    path = self.module_file(module)
    package = module if path and path.name == "__init__.py" else module.rsplit(".", 1)[0]
    parts = package.split(".")
    if node.level > 1:
      parts = parts[: len(parts) - (node.level - 1)]
    base = ".".join(parts)
    return f"{base}.{node.module}" if node.module else base

  def symbols(self, module: str) -> dict[str, tuple] | None:
    """Top-level names of a module.

    ("def", node) is a function or class. ("var", node) is an assignment.
    ("alias", module, name) comes from "from module import name".
    ("module", module) comes from "import module".
    """
    if module in self._symbols:
      return self._symbols[module]
    tree = self.tree(module)
    if tree is None:
      self._symbols[module] = None
      return None
    out: dict[str, tuple] = {}
    self._symbols[module] = out  # Stops import cycles.
    self._collect(module, tree.body, out)
    return out

  def _collect(self, module: str, body: list[ast.stmt], out: dict[str, tuple]) -> None:
    for node in body:
      if isinstance(node, (*FUNC_TYPES, ast.ClassDef)):
        out[node.name] = ("def", node)
      elif isinstance(node, ast.Assign):
        for target in node.targets:
          for name in target_names(target):
            out[name] = ("var", node)
      elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        out[node.target.id] = ("var", node)
      elif isinstance(node, ast.ImportFrom):
        source = self._absolute(module, node)
        for alias in node.names:
          if alias.name == "*":
            for name in self._public_names(source):
              out[name] = ("alias", source, name)
          else:
            out[alias.asname or alias.name] = ("alias", source, alias.name)
      elif isinstance(node, ast.Import):
        for alias in node.names:
          if alias.asname:
            out[alias.asname] = ("module", alias.name)
          else:
            top = alias.name.split(".")[0]
            out[top] = ("module", top)
      elif isinstance(node, ast.If):
        # Names imported only for type checkers do not exist at run time.
        if not is_type_checking(node.test):
          self._collect(module, node.body, out)
        self._collect(module, node.orelse, out)
      elif isinstance(node, ast.Try):
        self._collect(module, node.body, out)
        for handler in node.handlers:
          self._collect(module, handler.body, out)
        self._collect(module, node.orelse, out)
        self._collect(module, node.finalbody, out)

  def _public_names(self, module: str) -> list[str]:
    symbols = self.symbols(module)
    tree = self.tree(module)
    if not symbols or tree is None:
      return []
    for node in tree.body:
      if (
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
        and isinstance(node.value, (ast.List, ast.Tuple))
      ):
        return [
          e.value
          for e in node.value.elts
          if isinstance(e, ast.Constant) and isinstance(e.value, str)
        ]
    return [name for name in symbols if not name.startswith("_")]

  def resolve(self, module: str, name: str, depth: int = 0) -> tuple | None:
    """Find where "module.name" is defined.

    Returns ("def", module, node), ("var", module, node), ("module", module),
    ("unknown", module) when it cannot be checked, or None when it does not exist.
    """
    if depth > 25 or not self.owns(module):
      return ("unknown", module)
    symbols = self.symbols(module)
    if symbols is None:
      return None
    entry = symbols.get(name)
    if entry is not None:
      kind = entry[0]
      if kind == "def":
        return ("def", module, entry[1])
      if kind == "module":
        return self.module_ref(entry[1])
      if kind == "alias":
        source, original = entry[1], entry[2]
        if (source, original) == (module, name):
          # "from . import name" in a package imports its submodule.
          return self.module_ref(f"{source}.{original}")
        return self.resolve(source, original, depth + 1)
      if kind == "var":
        chain = attr_chain(getattr(entry[1], "value", None))
        if chain and chain != [name]:
          found, _ = self.walk(self.resolve(module, chain[0], depth + 1), chain[1:], depth + 1)
          if found is not None and found[0] in ("def", "module"):
            return found
        return ("var", module, entry[1])
    if self.module_file(f"{module}.{name}") is not None:
      return ("module", f"{module}.{name}")
    if "__getattr__" in symbols:
      return ("unknown", module)  # The module creates names on demand.
    return None

  def walk(self, found: tuple | None, attrs: list[str], depth: int = 0) -> tuple[tuple | None, int]:
    """Follow attributes such as "module.func" or "Class.Inner".

    Returns (result, count). If result is None, attrs[count - 1] was not found.
    If count is 0, the start was not found. Walking stops at functions,
    variables and unknown objects, because their attributes are not checked.
    """
    for index, attr in enumerate(attrs):
      if found is None:
        return None, index
      if found[0] == "module":
        found = self.resolve(found[1], attr, depth + 1)
      elif found[0] == "def" and isinstance(found[2], ast.ClassDef):
        member = self.class_members(found[1], found[2]).get(attr)
        if member is None:
          found = None
        else:
          module, node = member
          is_def = isinstance(node, (*FUNC_TYPES, ast.ClassDef))
          found = ("def", module, node) if is_def else ("var", module, node)
      else:
        return found, index
      if found is None:
        return None, index + 1
    return found, len(attrs)

  def resolve_expr(self, module: str, expr: ast.expr) -> tuple | None:
    if isinstance(expr, ast.Subscript):
      expr = expr.value
    chain = attr_chain(expr)
    if not chain:
      return None
    found, _ = self.walk(self.resolve(module, chain[0]), chain[1:])
    return found

  def class_members(self, module: str, cls: ast.ClassDef, depth: int = 0) -> dict[str, tuple]:
    """Names in a class body and in its base classes, as {name: (module, node)}."""
    key = (module, cls.name, cls.lineno)
    if key in self._members:
      return self._members[key]
    out: dict[str, tuple] = {}
    self._members[key] = out  # Stops cycles.
    if depth <= 10:
      for base in cls.bases:
        found = self.resolve_expr(module, base)
        if found is not None and found[0] == "def" and isinstance(found[2], ast.ClassDef):
          out.update(self.class_members(found[1], found[2], depth + 1))
    for node in cls.body:
      if isinstance(node, (*FUNC_TYPES, ast.ClassDef)):
        out[node.name] = (module, node)
      elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        out[node.target.id] = (module, node)
      elif isinstance(node, ast.Assign):
        for target in node.targets:
          for name in target_names(target):
            out[name] = (module, node)
    return out

  def is_dataclass(self, module: str, cls: ast.ClassDef, depth: int = 0) -> bool:
    if "dataclass" in decorator_names(cls):
      return True
    if depth > 10:
      return False
    for base in cls.bases:
      found = self.resolve_expr(module, base)
      if found is not None and found[0] == "def" and isinstance(found[2], ast.ClassDef):
        if self.is_dataclass(found[1], found[2], depth + 1):
          return True
    return False

  def signature(self, found: tuple | None, for_term: bool = False) -> Sig | None:
    """Parameters of a function or a class.

    With for_term=True, return the parameters that a manager term "params" dict
    fills: a function without its first parameter (env), or the "__call__" of a
    class without self and env.
    """
    if found is None or found[0] != "def":
      return None
    module, node = found[1], found[2]
    if isinstance(node, FUNC_TYPES):
      sig = func_sig(node, drop_first=False)
      return Sig(sig.names[1:], sig.var_kw) if for_term else sig
    if not isinstance(node, ast.ClassDef):
      return None
    members = self.class_members(module, node)
    if for_term:
      call = members.get("__call__")
      if call and isinstance(call[1], FUNC_TYPES):
        sig = func_sig(call[1], drop_first=True)
        return Sig(sig.names[1:], sig.var_kw)
      return None
    if self.is_dataclass(module, node):
      fields = [
        name
        for name, (_, value) in members.items()
        if isinstance(value, ast.AnnAssign) and not is_classvar(value)
      ]
      return Sig(fields, var_kw=False)
    init = members.get("__init__")
    if init and isinstance(init[1], FUNC_TYPES):
      return func_sig(init[1], drop_first=True)
    return None

  def data_members(self) -> set[str]:
    """Attribute names of every class whose name ends with "Data"."""
    if self._data_members is None:
      names: set[str] = set()
      for path in sorted(self.pkg_dir.rglob("*.py")):
        parts = list(path.relative_to(self.pkg_dir.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
          parts = parts[:-1]
        module = ".".join(parts)
        tree = self.tree(module)
        if tree is None:
          continue
        for node in tree.body:
          if isinstance(node, ast.ClassDef) and node.name.endswith("Data"):
            names.update(self.class_members(module, node))
      self._data_members = names
    return self._data_members


def read_dependencies(pyproject: Path) -> dict[str, str]:
  """Return {package: version spec} from the [project] dependencies list."""
  if not pyproject.exists():
    return {}
  text = pyproject.read_text(encoding="utf-8")
  section = re.search(r"^\[project\]\s*$(.*?)(?=^\[)", text, re.M | re.S)
  body = section.group(1) if section else text
  start = re.search(r"^\s*dependencies\s*=\s*\[", body, re.M)
  if not start:
    return {}
  items: list[str] = []
  buffer, in_string, i = "", False, start.end()
  while i < len(body):
    char = body[i]
    if in_string:
      if char == '"':
        items.append(buffer)
        buffer, in_string = "", False
      else:
        buffer += char
    elif char == '"':
      in_string = True
    elif char == "#":
      while i < len(body) and body[i] != "\n":
        i += 1
    elif char == "]":
      break
    i += 1
  deps: dict[str, str] = {}
  for item in items:
    match = re.match(r"\s*([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*(.*)", item)
    if match:
      deps[match.group(1).lower().replace("_", "-")] = match.group(3).strip() or "any"
  return deps


def read_uv_sources(pyproject: Path) -> dict[str, str]:
  """Return {package: source} from [tool.uv.sources], for example a git tag."""
  if not pyproject.exists():
    return {}
  text = pyproject.read_text(encoding="utf-8")
  section = re.search(r"^\[tool\.uv\.sources\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
  sources: dict[str, str] = {}
  if section:
    for line in section.group(1).splitlines():
      match = re.match(r"\s*([A-Za-z0-9_.\-]+)\s*=\s*\{(.*)\}", line)
      if match:
        sources[match.group(1).lower().replace("_", "-")] = match.group(2).strip()
  return sources


class RepoScan:
  """Scan the repo code and compare every package use with the old and new version."""

  def __init__(self, packages: dict[str, tuple[PackageIndex, PackageIndex]]):
    self.packages = packages
    self.findings: set[Finding] = set()
    self.unresolved: set[Finding] = set()
    self.stats = {"files": 0, "references": 0, "calls": 0, "classes": 0}

  def files(self):
    this_file = Path(__file__).resolve()
    for path in sorted(REPO_ROOT.rglob("*.py")):
      if any(part in SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts):
        continue
      if path.resolve() != this_file:
        yield path

  def scan_file(self, path: Path) -> None:
    rel = str(path.relative_to(REPO_ROOT))
    tree = ast.parse(path.read_text(encoding="utf-8"), rel)
    aliases = self._aliases(tree, rel)
    if aliases:
      self.stats["files"] += 1
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
      if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
        if node.value.attr == "data":
          self._check_data_attr(node, rel)
      if not aliases:
        continue
      if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
        parent = parents.get(node)
        if not (isinstance(parent, ast.Attribute) and parent.value is node):
          self._check_reference(node, aliases, rel)
      if isinstance(node, ast.Call):
        self._check_call(node, aliases, rel)
      if isinstance(node, ast.ClassDef):
        self._check_class(node, aliases, rel)

  def _package_of(self, module: str) -> str | None:
    top = module.split(".")[0]
    return top if top in self.packages else None

  def _aliases(self, tree: ast.Module, rel: str) -> dict[str, tuple]:
    """Local names bound to package modules or package names, from every import."""
    out: dict[str, tuple] = {}
    for node in ast.walk(tree):
      if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        pkg = self._package_of(node.module)
        if pkg is None:
          continue
        for alias in node.names:
          if alias.name == "*":
            continue
          out[alias.asname or alias.name] = ("name", pkg, node.module, alias.name)
          self._check_import(pkg, node.module, alias.name, rel, node.lineno)
      elif isinstance(node, ast.Import):
        for alias in node.names:
          pkg = self._package_of(alias.name)
          if pkg is None:
            continue
          self._check_import(pkg, alias.name, None, rel, node.lineno)
          if alias.asname:
            out[alias.asname] = ("module", pkg, alias.name)
          else:
            top = alias.name.split(".")[0]
            out[top] = ("module", pkg, top)
    return out

  @staticmethod
  def _start(index: PackageIndex, entry: tuple) -> tuple | None:
    if entry[0] == "module":
      return index.module_ref(entry[2])
    return index.resolve(entry[2], entry[3])

  @staticmethod
  def _full_name(entry: tuple, attrs: list[str]) -> str:
    base = entry[2] if entry[0] == "module" else f"{entry[2]}.{entry[3]}"
    return ".".join([base, *attrs])

  def _record_missing(self, old_found, api: str, rel: str, line: int, what: str) -> None:
    if old_found is not None:
      self.findings.add(Finding("missing", rel, line, api, f"{what} exists in old, not in new"))
    else:
      self.unresolved.add(Finding("unresolved", rel, line, api, f"{what} not found in old or new"))

  def _check_import(self, pkg: str, module: str, name: str | None, rel: str, line: int) -> None:
    old, new = self.packages[pkg]
    if name is None:
      old_found, new_found, api = old.module_ref(module), new.module_ref(module), module
    else:
      old_found, new_found = old.resolve(module, name), new.resolve(module, name)
      api = f"{module}.{name}"
    if new_found is None:
      self._record_missing(old_found, api, rel, line, "import")

  def _check_reference(self, node: ast.expr, aliases: dict, rel: str) -> None:
    chain = attr_chain(node)
    if not chain or chain[0] not in aliases:
      return
    entry = aliases[chain[0]]
    old, new = self.packages[entry[1]]
    self.stats["references"] += 1
    new_found, count = new.walk(self._start(new, entry), chain[1:])
    if new_found is not None or count == 0:
      return  # Found, or the import itself is already reported.
    old_found, _ = old.walk(self._start(old, entry), chain[1 : count + 1])
    api = self._full_name(entry, chain[1 : count + 1])
    self._record_missing(old_found, api, rel, node.lineno, "attribute")

  def _callable(self, expr: ast.expr, aliases: dict) -> tuple | None:
    """Return (api name, old found, new found, old index, new index) for a package callable."""
    chain = attr_chain(expr)
    if not chain or chain[0] not in aliases:
      return None
    entry = aliases[chain[0]]
    old, new = self.packages[entry[1]]
    old_found, _ = old.walk(self._start(old, entry), chain[1:])
    new_found, _ = new.walk(self._start(new, entry), chain[1:])
    return self._full_name(entry, chain[1:]), old_found, new_found, old, new

  def _check_call(self, node: ast.Call, aliases: dict, rel: str) -> None:
    target = self._callable(node.func, aliases)
    if target is not None:
      api, old_found, new_found, old, new = target
      new_sig, old_sig = new.signature(new_found), old.signature(old_found)
      if new_sig is not None:
        self.stats["calls"] += 1
        for kw in node.keywords:
          if kw.arg and kw.arg not in new_sig.names and not new_sig.var_kw:
            self._record_argument(old_sig, kw.arg, "argument", api, rel, node.lineno)
    # Manager term configs pass "params" to "func": func=<callable>, params={...}.
    kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
    params = kwargs.get("params")
    if "func" not in kwargs or not isinstance(params, ast.Dict):
      return
    target = self._callable(kwargs["func"], aliases)
    if target is None:
      return
    api, old_found, new_found, old, new = target
    new_sig = new.signature(new_found, for_term=True)
    old_sig = old.signature(old_found, for_term=True)
    if new_sig is None:
      return
    for key in params.keys:
      if isinstance(key, ast.Constant) and isinstance(key.value, str):
        if key.value not in new_sig.names and not new_sig.var_kw:
          self._record_argument(old_sig, key.value, "params key", api, rel, key.lineno)

  def _record_argument(self, old_sig, name, what, api, rel, line) -> None:
    accepted_before = old_sig is not None and (name in old_sig.names or old_sig.var_kw)
    detail = f"{what} '{name}' is not accepted in new"
    if accepted_before:
      detail += "; it was accepted in old"
    self.findings.add(Finding("argument", rel, line, api, detail))

  def _check_class(self, node: ast.ClassDef, aliases: dict, rel: str) -> None:
    own = {n.name: n for n in node.body if isinstance(n, FUNC_TYPES)}
    for base in node.bases:
      target = self._callable(base.value if isinstance(base, ast.Subscript) else base, aliases)
      if target is None:
        continue
      base_name, old_found, new_found, old, new = target
      if not (new_found and new_found[0] == "def" and isinstance(new_found[2], ast.ClassDef)):
        continue
      self.stats["classes"] += 1
      new_members = new.class_members(new_found[1], new_found[2])
      old_members = {}
      if old_found and old_found[0] == "def" and isinstance(old_found[2], ast.ClassDef):
        old_members = old.class_members(old_found[1], old_found[2])
      for name, method in own.items():
        mine = func_sig(method, drop_first=True)
        new_base = new_members.get(name)
        old_base = old_members.get(name)
        old_sig = func_sig(old_base[1], True) if old_base and isinstance(old_base[1], FUNC_TYPES) else None
        if new_base is None:
          if old_sig is not None:
            detail = f"{base_name}.{name} exists in old, not in new; it will not be called"
            self.findings.add(Finding("override", rel, method.lineno, f"{node.name}.{name}", detail))
          continue
        if not isinstance(new_base[1], FUNC_TYPES):
          continue
        new_sig = func_sig(new_base[1], True)
        if mine.names != new_sig.names and (old_sig is None or old_sig.names != new_sig.names):
          detail = (
            f"repo ({', '.join(mine.names)}) but new {base_name}.{name} "
            f"({', '.join(new_sig.names)})"
          )
          self.findings.add(Finding("override", rel, method.lineno, f"{node.name}.{name}", detail))
      for name, (_, member) in new_members.items():
        if name in own or not isinstance(member, FUNC_TYPES):
          continue
        if "abstractmethod" not in decorator_names(member):
          continue
        old_member = old_members.get(name)
        was_abstract = bool(old_member) and "abstractmethod" in decorator_names(old_member[1])
        if not was_abstract:
          detail = f"new {base_name} has abstract method '{name}'"
          self.findings.add(Finding("abstract", rel, node.lineno, node.name, detail))

  def _check_data_attr(self, node: ast.Attribute, rel: str) -> None:
    for pkg, (old, new) in self.packages.items():
      if node.attr in old.data_members() and node.attr not in new.data_members():
        detail = f"in an old {pkg} *Data class, in no new one"
        self.findings.add(Finding("data", rel, node.lineno, f".data.{node.attr}", detail))


TITLES = {
  "missing": "2. Names not found in the new version",
  "argument": "3. Arguments not accepted by the new version",
  "override": "4. Overridden methods with a changed base method",
  "abstract": "5. New abstract methods",
  "data": "6. Data attributes not found in the new version",
  "unresolved": "Not checked: found in neither version",
}


def print_dependencies(packages: dict[str, tuple[PackageIndex, PackageIndex]]) -> None:
  repo_pyproject = REPO_ROOT / "pyproject.toml"
  repo_deps = read_dependencies(repo_pyproject)
  repo_sources = read_uv_sources(repo_pyproject)
  print("## 1. Dependencies\n")
  for pkg, (old, new) in packages.items():
    old_deps = read_dependencies(old.source / "pyproject.toml")
    new_deps = read_dependencies(new.source / "pyproject.toml")
    print(f"### {pkg} {old.version()} -> {new.version()}\n")
    pin = repo_sources.get(pkg.replace("_", "-"))
    if pin:
      print(f"This repo pins {pkg}: `{pin}`\n")
    print("| package | old requires | new requires | this repo |")
    print("|---|---|---|---|")
    for dep in sorted(set(old_deps) | set(new_deps)):
      old_spec, new_spec = old_deps.get(dep, "-"), new_deps.get(dep, "-")
      if old_spec == new_spec:
        continue
      mine = repo_deps.get(dep, "-")
      if dep in repo_sources:
        mine += f" `{repo_sources[dep]}`"
      print(f"| {dep} | {old_spec} | {new_spec} | {mine} |")
    print()


def print_findings(check: str, findings: set[Finding]) -> None:
  rows: dict[tuple, list[int]] = defaultdict(list)
  for f in findings:
    if f.check == check:
      rows[(f.file, f.api, f.detail)].append(f.line)
  print(f"## {TITLES[check]}\n")
  if not rows:
    print("None.\n")
    return
  print("| file | lines | API | detail |")
  print("|---|---|---|---|")
  for (file, api, detail), lines in sorted(rows.items()):
    line_text = ", ".join(str(n) for n in sorted(set(lines)))
    print(f"| {file} | {line_text} | `{api}` | {detail} |")
  print()


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
    "--pkg",
    nargs=3,
    action="append",
    required=True,
    metavar=("NAME", "OLD_SOURCE", "NEW_SOURCE"),
    help="package name, old source folder, new source folder",
  )
  args = parser.parse_args()
  packages = {
    name: (PackageIndex(name, Path(old)), PackageIndex(name, Path(new)))
    for name, old, new in args.pkg
  }
  scan = RepoScan(packages)
  for path in scan.files():
    scan.scan_file(path)

  versions = ", ".join(f"{n} {o.version()} -> {w.version()}" for n, (o, w) in packages.items())
  print(f"# API check: {versions}\n")
  s = scan.stats
  print(
    f"Scanned {s['files']} files that import these packages: {s['references']} references, "
    f"{s['calls']} calls with known signatures, {s['classes']} subclasses.\n"
  )
  print_dependencies(packages)
  for check in ("missing", "argument", "override", "abstract", "data"):
    print_findings(check, scan.findings)
  print_findings("unresolved", scan.unresolved)
  counts: dict[str, int] = defaultdict(int)
  for f in scan.findings:
    counts[f.file] += 1
  print("## Findings per file\n")
  print("| file | findings |")
  print("|---|---|")
  for file, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
    print(f"| {file} | {count} |")


if __name__ == "__main__":
  main()
