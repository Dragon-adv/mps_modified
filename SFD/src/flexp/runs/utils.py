import os
import os.path
import shutil
import inspect
import functools
from typing import Any, Callable, TypeVar, cast, overload, Iterable, Union
from collections import UserDict
from datetime import datetime

import global_config


F = TypeVar('F', bound=Callable)


class RunRecord(UserDict[str, Any]):
    @overload
    def __init__(self, *, print_type_args: Union[str, Iterable[str]] = ()): ...

    @overload
    def __init__(
        self,
        func: Callable,
        arguments: dict[str, Any],
        *,
        print_type_args: Union[str, Iterable[str]] = ()
    ): ...

    def __init__(self, func=None, arguments=None, *, print_type_args: Iterable[str] = ()):
        super().__init__(arguments)
        self.func = cast(Callable, func)
        self.start_time = datetime.now()
        self.print_type_args = (
            {print_type_args} if isinstance(print_type_args, str)
            else set(print_type_args)
        )
        """only print the type of these arguments, not the value"""

    @property
    def arguments(self) -> dict[str, Any]:
        return self.data

    def __str__(self) -> str:
        timestamp = self.start_time.strftime('%y-%m-%d %H:%M:%S')
        sb = [
            f'[{self.func.__module__}.{self.func.__qualname__}] {timestamp}'
        ]
        for name, value in self.arguments.items():
            if name not in self.print_type_args or value is None or isinstance(value, (int, float, str, bool)):
                sb.append(f'{name}: {value!r}')
            else:
                arg_type = type(value)
                full_class_name = f'{arg_type.__module__}.{arg_type.__qualname__}'
                sb.append(f'{name}: (type) {full_class_name}')
        return '\n'.join(sb)

    @staticmethod
    def record(func: F) -> F:
        @functools.wraps(func)
        def func_wrapper(*args, **kwargs):
            (record_param_name, record_param_default) = next((
                (name, param.default)
                for name, param in inspect.signature(func).parameters.items()
                if isinstance(param.default, RunRecord)
            ), (None, None))
            if record_param_name is not None and record_param_default is not None:
                bound_args = inspect.signature(func).bind(*args, **kwargs)
                bound_args.apply_defaults()
                arguments = {
                    name: value for name, value in bound_args.arguments.items()
                    if name != record_param_name
                }
                return func(
                    *args,
                    **kwargs,
                    **{record_param_name: RunRecord(func, arguments, print_type_args=record_param_default.print_type_args)}
                )
            else:
                return func(*args, **kwargs)
    
        return cast(F, func_wrapper)


def keep_with_confirm(folder: str):
    while True:
        option = input(f'Do you want to keep the result {os.path.abspath(folder)} [y/n]? ').strip()
        if option == 'y':
            return
        elif option == 'n':
            break
    shutil.rmtree(folder)


def prepare_result_folder(name: str, start_time, record_text: str) -> str:
    timestamp = start_time.strftime('%y-%m-%d %H-%M-%S')
    result_folder = os.path.join(global_config.RESULT_ROOT, f'[{timestamp}]{name}')
    os.makedirs(result_folder, exist_ok=True)
    with open(os.path.join(result_folder, 'run_record.txt'), 'w', encoding='utf8') as file:
        print(record_text, file=file)
    return result_folder
