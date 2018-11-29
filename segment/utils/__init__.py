import pkgutil

for importer, modname, ispkg in pkgutil.walk_packages(path=__path__, onerror=lambda x: None):
    importer.find_module(modname).load_module(modname)