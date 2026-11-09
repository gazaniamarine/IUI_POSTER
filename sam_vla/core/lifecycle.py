from typing import List, Protocol


class Service(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...


class ServiceRegistry:
    def __init__(self) -> None:
        self._services: List[Service] = []

    def register(self, service: Service) -> None:
        self._services.append(service)

    def start_all(self) -> None:
        for service in self._services:
            try:
                service.start()
            except Exception as exc:
                print(f"[ServiceRegistry] {service!r} failed to start: {exc}")

    def stop_all(self) -> None:
        for service in reversed(self._services):
            try:
                service.stop()
            except Exception as exc:
                print(f"[ServiceRegistry] {service!r} failed to stop: {exc}")


if __name__ == "__main__":

    class ServiceA:
        def start(self) -> None:
            print("A started")

        def stop(self) -> None:
            print("A stopped")

    class ServiceB:
        def start(self) -> None:
            print("B started")

        def stop(self) -> None:
            print("B stopped")

    registry = ServiceRegistry()
    registry.register(ServiceA())
    registry.register(ServiceB())

    registry.start_all()
    registry.stop_all()
