from __future__ import annotations

import pygame
from dataclasses import dataclass
from geometry.point import Point
from config import screen_width as V_WIDTH, screen_height as V_HEIGHT

@dataclass(frozen=True)
class ViewportTransform:
    scale: float
    offset_x: int
    offset_y: int
    width: int
    height: int

    def map_window_to_virtual(
        self,
        x: int,
        y: int,
        virtual_width: int | None = None,
        virtual_height: int | None = None,
    ) -> tuple[int, int] | None:
        if virtual_width is None:
            virtual_width = V_WIDTH
        if virtual_height is None:
            virtual_height = V_HEIGHT

        within_x = self.offset_x <= x < (self.offset_x + self.width)
        within_y = self.offset_y <= y < (self.offset_y + self.height)
        if not (within_x and within_y):
            return None

        virtual_x = int((x - self.offset_x) / self.scale)
        virtual_y = int((y - self.offset_y) / self.scale)

        # Clamp to provided virtual bounds.
        virtual_x = min(max(virtual_x, 0), virtual_width - 1)
        virtual_y = min(max(virtual_y, 0), virtual_height - 1)
        return (virtual_x, virtual_y)


def get_viewport_transform(
    window_width: int,
    window_height: int,
    virtual_width: int,
    virtual_height: int,
) -> ViewportTransform:
    """
    Utility function used by unit tests.

    Computes a letterboxed transform that maps a window coordinate space
    (window_width x window_height) into a virtual coordinate space
    (virtual_width x virtual_height).
    """
    scale = min(window_width / virtual_width, window_height / virtual_height)
    viewport_width = int(virtual_width * scale)
    viewport_height = int(virtual_height * scale)
    offset_x = (window_width - viewport_width) // 2
    offset_y = (window_height - viewport_height) // 2

    return ViewportTransform(
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
        width=viewport_width,
        height=viewport_height,
    )

class Viewport:
    def __init__(self, mediator):
        self.mediator = mediator
        self.virtual_size = (V_WIDTH, V_HEIGHT)

    def _get_transform(self, window_width, window_height) -> ViewportTransform:
        v_w, v_h = self.virtual_size
        scale = min(window_width / v_w, window_height / v_h)
        viewport_width = int(v_w * scale)
        viewport_height = int(v_h * scale)
        offset_x = (window_width - viewport_width) // 2
        offset_y = (window_height - viewport_height) // 2
        
        return ViewportTransform(
            scale=scale, offset_x=offset_x, offset_y=offset_y,
            width=viewport_width, height=viewport_height
        )

    def draw(self, surface: pygame.Surface, time_ms: int):
        # 1. 가상 1920x1080 서피스 생성 및 렌더링
        temp_surface = pygame.Surface(self.virtual_size)
        temp_surface.fill((255, 255, 255))
        self.mediator.render(temp_surface)

        # 2. 현재 창 크기에 맞춘 트랜스폼 계산
        win_w, win_h = surface.get_size()
        transform = self._get_transform(win_w, win_h)

        # 3. 비율 유지 스케일링 (Smooth Scale)
        scaled_surface = pygame.transform.smoothscale(temp_surface, (transform.width, transform.height))
        
        # 4. 중앙 정렬 (Letterboxing)
        surface.blit(scaled_surface, (transform.offset_x, transform.offset_y))

    def map_window_to_virtual(self, x, y, window_width, window_height):
        transform = self._get_transform(window_width, window_height)
        v_w, v_h = self.virtual_size
        mapped = transform.map_window_to_virtual(x, y, virtual_width=v_w, virtual_height=v_h)
        if mapped:
            return Point(mapped[0], mapped[1])
        return None