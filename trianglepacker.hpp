#ifndef _H_TRIANGLEPACKER_H_
#define _H_TRIANGLEPACKER_H_

#include <vector> // std::vector
#include <memory> // std::unique_ptr
#include <string> // std::to_string

namespace ray
{
	namespace detail
	{
		class exception : public std::exception
		{
		public:
			const int id;

			virtual const char* what() const noexcept override
			{
				return m.what();
			}

		protected:
			exception(int id_, const char* what_arg)
				: id(id_), m(what_arg)
			{}

			static std::string name(const std::string& ename, int id)
			{
				return "[uvmapper.exception." + ename + "." + std::to_string(id) + "] ";
			}

		private:
			std::runtime_error m;
		};

		class out_of_range : public exception
		{
		public:
			static out_of_range create(int id, const std::string& what_arg)
			{
				std::string w = exception::name("out_of_range", id) + what_arg;
				return out_of_range(id, w.c_str());
			}

		private:
			out_of_range(int id_, const char* what_arg)
				: exception(id_, what_arg)
			{
			}
		};
		template<typename T>
		struct Vector2
		{
			T x, y;

			Vector2() = default;
			Vector2(T xx, T yy) noexcept :x(xx), y(yy) {}

			Vector2& operator+=(T scale) noexcept { x += scale; y += scale; return *this; }
			Vector2& operator-=(T scale) noexcept { x -= scale; y -= scale; return *this; }
			Vector2& operator*=(T scale) noexcept { x *= scale; y *= scale; return *this; }
			Vector2& operator/=(T scale) noexcept { x /= scale; y /= scale; return *this; }

			Vector2& operator+=(const Vector2& r) noexcept { x += r.x; y += r.y; return *this; }
			Vector2& operator-=(const Vector2& r) noexcept { x -= r.x; y -= r.y; return *this; }
			Vector2& operator*=(const Vector2& r) noexcept { x *= r.x; y *= r.y; return *this; }
			Vector2& operator/=(const Vector2& r) noexcept { x /= r.x; y /= r.y; return *this; }
		};

		template<typename T>
		struct Vector3
		{
			T x, y, z;

			Vector3() = default;
			Vector3(T xx, T yy, T zz) noexcept :x(xx), y(yy), z(zz) {}

			Vector3& operator+=(const Vector3& r) noexcept { x += r.x; y += r.y; z += r.z; return *this; }
			Vector3& operator-=(const Vector3& r) noexcept { x -= r.x; y -= r.y; z -= r.z; return *this; }
			Vector3& operator*=(const Vector3& r) noexcept { x *= r.x; y *= r.y; z *= r.z; return *this; }
			Vector3& operator/=(const Vector3& r) noexcept { x /= r.x; y /= r.y; z /= r.z; return *this; }
		};

		template<typename T>
		struct Vector4
		{
			T x, y, z, w;

			Vector4() = default;
			Vector4(T xx, T yy, T zz, T ww) noexcept :x(xx), y(yy), z(zz), w(ww) {}

			Vector4& operator+=(const Vector4& r) noexcept { x += r.x; y += r.y; z += r.z; w += r.w; return *this; }
			Vector4& operator-=(const Vector4& r) noexcept { x -= r.x; y -= r.y; z -= r.z; w -= r.w; return *this; }
			Vector4& operator*=(const Vector4& r) noexcept { x *= r.x; y *= r.y; z *= r.z; w *= r.w; return *this; }
			Vector4& operator/=(const Vector4& r) noexcept { x /= r.x; y /= r.y; z /= r.z; w /= r.w; return *this; }
		};

		template<typename T>
		inline constexpr Vector2<T> operator+(const Vector2<T>& l, T scale) noexcept { return Vector2<T>(l.x + scale, l.y + scale); }
		template<typename T>
		inline constexpr Vector2<T> operator-(const Vector2<T>& l, T scale) noexcept { return Vector2<T>(l.x - scale, l.y - scale); }
		template<typename T>
		inline constexpr Vector2<T> operator*(const Vector2<T>& l, T scale) noexcept { return Vector2<T>(l.x * scale, l.y * scale); }
		template<typename T>
		inline constexpr Vector2<T> operator/(const Vector2<T>& l, T scale) noexcept { return Vector2<T>(l.x / scale, l.y / scale); }

		template<typename T>
		inline constexpr Vector2<T> operator+(const Vector2<T>& l, const Vector2<T>& r) noexcept { return Vector2<T>(l.x + r.x, l.y + r.y); }
		template<typename T>
		inline constexpr Vector2<T> operator-(const Vector2<T>& l, const Vector2<T>& r) noexcept { return Vector2<T>(l.x - r.x, l.y - r.y); }
		template<typename T>
		inline constexpr Vector2<T> operator*(const Vector2<T>& l, const Vector2<T>& r) noexcept { return Vector2<T>(l.x * r.x, l.y * r.y); }
		template<typename T>
		inline constexpr Vector2<T> operator/(const Vector2<T>& l, const Vector2<T>& r) noexcept { return Vector2<T>(l.x / r.x, l.y / r.y); }

		template<typename T>
		inline constexpr Vector3<T> operator+(const Vector3<T>& l, T scale) noexcept { return Vector3<T>(l.x + scale, l.y + scale, l.z + scale); }
		template<typename T>
		inline constexpr Vector3<T> operator-(const Vector3<T>& l, T scale) noexcept { return Vector3<T>(l.x - scale, l.y - scale, l.z - scale); }
		template<typename T>
		inline constexpr Vector3<T> operator*(const Vector3<T>& l, T scale) noexcept { return Vector3<T>(l.x * scale, l.y * scale, l.z * scale); }
		template<typename T>
		inline constexpr Vector3<T> operator/(const Vector3<T>& l, T scale) noexcept { return Vector3<T>(l.x / scale, l.y / scale, l.z / scale); }

		template<typename T>
		inline constexpr Vector3<T> operator+(const Vector3<T>& l, const Vector3<T>& r) noexcept { return Vector3<T>(l.x + r.x, l.y + r.y, l.z + r.z); }
		template<typename T>
		inline constexpr Vector3<T> operator-(const Vector3<T>& l, const Vector3<T>& r) noexcept { return Vector3<T>(l.x - r.x, l.y - r.y, l.z - r.z); }
		template<typename T>
		inline constexpr Vector3<T> operator*(const Vector3<T>& l, const Vector3<T>& r) noexcept { return Vector3<T>(l.x * r.x, l.y * r.y, l.z * r.z); }
		template<typename T>
		inline constexpr Vector3<T> operator/(const Vector3<T>& l, const Vector3<T>& r) noexcept { return Vector3<T>(l.x / r.x, l.y / r.y, l.z / r.z); }

		template<typename T>
		inline constexpr Vector4<T> operator+(const Vector4<T>& l, const Vector4<T>& r) noexcept { return Vector4<T>(l.x + r.x, l.y + r.y, l.z + r.z, l.w + r.w); }
		template<typename T>
		inline constexpr Vector4<T> operator-(const Vector4<T>& l, const Vector4<T>& r) noexcept { return Vector4<T>(l.x - r.x, l.y - r.y, l.z - r.z, l.w - r.w); }
		template<typename T>
		inline constexpr Vector4<T> operator*(const Vector4<T>& l, const Vector4<T>& r) noexcept { return Vector4<T>(l.x * r.x, l.y * r.y, l.z * r.z, l.w * r.w); }
		template<typename T>
		inline constexpr Vector4<T> operator/(const Vector4<T>& l, const Vector4<T>& r) noexcept { return Vector4<T>(l.x / r.x, l.y / r.y, l.z / r.z, l.w / r.w); }

		template<typename T>
		inline constexpr T dot(const Vector2<T>& v1, const Vector2<T>& v2) noexcept { return v1.x * v2.x + v1.y * v2.y; }
		template<typename T>
		inline constexpr T dot(const Vector3<T>& v1, const Vector3<T>& v2) noexcept { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }

		template<typename T>
		inline constexpr T length2(const Vector2<T>& v) noexcept { return dot(v, v); }
		template<typename T>
		inline constexpr T length2(const Vector3<T>& v) noexcept { return dot(v, v); }

		template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
		inline constexpr T length(const Vector2<T>& v) noexcept { return std::sqrt(length2(v)); }
		template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
		inline constexpr T length(const Vector3<T>& v) noexcept { return std::sqrt(length2(v)); }

		template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
		inline constexpr Vector2<T> normalize(const Vector2<T>& v) noexcept
		{
			T magSq = length2(v);
			if (magSq > 0.0f)
			{
				T invSqrt = 1.0f / std::sqrt(magSq);
				return v * invSqrt;
			}

			return v;
		}

		template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
		inline constexpr Vector3<T> normalize(const Vector3<T>& v) noexcept
		{
			T magSq = length2(v);
			if (magSq > 0.0f)
			{
				T invSqrt = 1.0f / std::sqrt(magSq);
				return v * invSqrt;
			}

			return v;
		}

		template<typename _Tx, typename _Ty = int>
		struct Triangle
		{
			//       C           -
			//     * |  *        | h
			//   *   |     *     |
			// B-----+--------A  -
			// '--x--'        |
			// '-------w------'
			_Ty w, h, x;

			Vector2<_Tx> uv[3];

			Triangle() = default;
			Triangle(_Ty ww, _Ty hh, _Ty xx) noexcept : w(ww), h(hh), x(xx) { std::memset(this->uv, 0, sizeof(this->uv)); }
			Triangle(Vector3<_Tx> v1, Vector3<_Tx> v2, Vector3<_Tx> v3) noexcept { this->compute(v1, v2, v3); }

			constexpr auto area() const noexcept
			{
				return w * h;
			}

			void compute(Vector3<_Tx> v1, Vector3<_Tx> v2, Vector3<_Tx> v3) noexcept
			{
				Vector3<_Tx> tv[3];
				tv[0] = v2 - v1;
				tv[1] = v3 - v2;
				tv[2] = v1 - v3;

				_Tx len2[3];
				len2[0] = length2(tv[0]);
				len2[1] = length2(tv[1]);
				len2[2] = length2(tv[2]);

				std::uint8_t maxi; _Tx maxl = len2[0]; maxi = 0;
				if (len2[1] > maxl) { maxl = len2[1]; maxi = 1; }
				if (len2[2] > maxl) { maxl = len2[2]; maxi = 2; }
				std::uint8_t nexti = (maxi + 1) % 3;

				_Tx ww = std::sqrt(maxl);
				_Tx xx = -dot(tv[maxi], tv[nexti]) / ww;
				_Tx hh = length((tv[maxi] + tv[nexti]) - normalize(tv[maxi]) * (ww - xx));

				this->w = std::ceil(ww);
				this->x = std::ceil(xx);
				this->h = std::ceil(hh);

				std::memset(this->uv, 0, sizeof(this->uv));
			}
		};

		template<typename _Tx, typename _Ty, typename std::enable_if_t<std::is_pointer_v<_Ty>>* = nullptr>
		struct Quad
		{
			_Ty t1;
			_Ty t2;

			Vector2<_Tx> edge;

			Quad() noexcept : t1(nullptr), t2(nullptr) {}
			Quad(_Ty _t1, _Ty _t2) noexcept : t1(_t1), t2(_t2) { this->computeEdge(); }

			constexpr auto area() const noexcept
			{
				return edge.x * edge.y;
			}

			void computeEdge() noexcept
			{
				if (t1 && t2)
				{
					float area = sqrt((t1->area() + t2->area()) * 0.5f);
					edge.x = area;
					edge.y = area;
				}
				else
				{
					if (t1)
					{
						float area = sqrt(t1->area());
						edge.x = area;
						edge.y = area;
					}

					if (t2)
					{
						float area = sqrt(t2->area());
						edge.x = area;
						edge.y = area;
					}
				}
			}

			void computeEdge(_Ty _t1, _Ty _t2) noexcept
			{
				t1 = _t1;
				t2 = _t2;
				this->computeEdge();
			}

			void computeUV(_Tx margin = 0.0f) noexcept
			{
				if (t1)
				{
					t1->uv[0] = Vector2<_Tx>(0.0f, margin);
					t1->uv[1] = Vector2<_Tx>(0.0f, edge.y);
					t1->uv[2] = Vector2<_Tx>(edge.x - margin, edge.y);
				}

				if (t2)
				{
					t2->uv[0] = Vector2<_Tx>(margin, 0.0f);
					t2->uv[1] = Vector2<_Tx>(edge.x, edge.y - margin);
					t2->uv[2] = Vector2<_Tx>(edge.x, 0.0f);
				}
			}
		};

		template<typename _Tx, typename _Ty, typename std::enable_if_t<std::is_pointer_v<_Ty>>* = nullptr>
		class QuadNode
		{
		public:
			QuadNode() noexcept : _rect(0.0f, 0.0f, 1.0f, 1.0f) {}
			QuadNode(const QuadNode&) = delete;
			QuadNode& operator=(QuadNode&) = delete;

			QuadNode* insert(const Quad<_Tx, _Ty>& q, const Vector2<_Tx>& margin, bool write = true)
			{
				if (_child[0] && _child[1])
				{
					QuadNode<_Tx, _Ty>* c = _child[0]->insert(q, margin);
					return c ? c : _child[1]->insert(q, margin);
				}
				else
				{
					if (q.edge.x > _rect.z || q.edge.y > _rect.w)
						return nullptr;

					if (q.edge.x == _rect.z && q.edge.y == _rect.w)
					{
						if (write)
						{
							Vector2<_Tx> offset(_rect.x, _rect.y);
							if (q.t1)
							{
								q.t1->uv[0] += offset;
								q.t1->uv[1] += offset;
								q.t1->uv[2] += offset;
							}

							if (q.t2)
							{
								q.t2->uv[0] += offset;
								q.t2->uv[1] += offset;
								q.t2->uv[2] += offset;
							}
						}

						return this;
					}

					_child[0] = std::make_unique<QuadNode<_Tx, _Ty>>();
					_child[1] = std::make_unique<QuadNode<_Tx, _Ty>>();

					_Tx dw = _rect.z - q.edge.x;
					_Tx dh = _rect.w - q.edge.y;

					if (dw < dh)
					{
						_child[0]->_rect = Vector4<_Tx>(_rect.x + q.edge.x, _rect.y, _rect.z - q.edge.x, q.edge.y);
						_child[1]->_rect = Vector4<_Tx>(_rect.x, _rect.y + q.edge.y, _rect.z, _rect.w - q.edge.y);

						_child[0]->_rect += Vector4<_Tx>(margin.x, 0.f, -margin.x, 0.f);
						_child[1]->_rect += Vector4<_Tx>(0.f, margin.y, 0.f, -margin.y);
					}
					else
					{
						_child[0]->_rect = Vector4<_Tx>(_rect.x, _rect.y + q.edge.y, q.edge.x, _rect.w - q.edge.y);
						_child[1]->_rect = Vector4<_Tx>(_rect.x + q.edge.x, _rect.y, _rect.z - q.edge.x, _rect.w);

						_child[0]->_rect += Vector4<_Tx>(0.f, margin.y, 0.f, -margin.y);
						_child[1]->_rect += Vector4<_Tx>(margin.x, 0.f, -margin.x, 0.f);
					}

					if (write)
					{
						Vector2<_Tx> offset(_rect.x, _rect.y);
						if (q.t1)
						{
							q.t1->uv[0] += offset;
							q.t1->uv[1] += offset;
							q.t1->uv[2] += offset;
						}

						if (q.t2)
						{
							q.t2->uv[0] += offset;
							q.t2->uv[1] += offset;
							q.t2->uv[2] += offset;
						}
					}

					return this;
				}
			}

		private:
			Vector4<_Tx> _rect;
			std::unique_ptr<QuadNode<_Tx, _Ty>> _child[2];
		};
	}

	template<typename T, typename std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
	class basic_uvmapper
	{
	public:
		using value_t = T;

		using size_type = std::size_t;

		using vec2_t = detail::Vector2<value_t>;
		using vec3_t = detail::Vector3<value_t>;
		using vec4_t = detail::Vector4<value_t>;

		using triangle_t = detail::Triangle<value_t>;

		using quad_t = detail::Quad<value_t, triangle_t*>;
		using quad_node_t = detail::QuadNode<value_t, triangle_t*>;

		using exception = detail::exception;
		using out_of_range = detail::out_of_range;

		basic_uvmapper() = default;
		basic_uvmapper(const basic_uvmapper&) = delete;
		basic_uvmapper& operator=(basic_uvmapper&) = delete;

		static bool lightmappack(const value_t* positions, size_type vertexCount, int width, int height, value_t scale, value_t scalefactor, int margin, value_t* outUVs)
		{
			const vec3_t* p = (const vec3_t*)positions;

			std::vector<triangle_t> tris(vertexCount / 3);

			for (size_type i = 0; i < tris.size(); i++)
			{
				vec3_t tp[3];
				tp[0] = p[i * 3 + 0] * scale;
				tp[1] = p[i * 3 + 1] * scale;
				tp[2] = p[i * 3 + 2] * scale;

				tris[i].compute(tp[0], tp[1], tp[2]);
			}

			return lightmappack(tris, width, height, scale, scalefactor, margin, outUVs) == tris.size();
		}

		static bool lightmappack(const value_t* positions, size_type vertexCount, int width, int height, value_t scale, int margin, value_t* outUVs)
		{
			const vec3_t* p = (const vec3_t*)positions;

			std::vector<triangle_t> tris(vertexCount / 3);

			for (size_type i = 0; i < tris.size(); i++)
			{
				vec3_t tp[3];
				tp[0] = p[i * 3 + 0] * scale;
				tp[1] = p[i * 3 + 1] * scale;
				tp[2] = p[i * 3 + 2] * scale;

				tris[i].compute(tp[0], tp[1], tp[2]);
			}

			return lightmappack(tris, width, height, scale, margin, outUVs) == tris.size();
		}

		template<typename index_t = short>
		static bool lightmappack(const value_t* positions, const index_t* indices, size_type indexCount, int width, int height, value_t scale, int margin, value_t* outVertices, value_t* outUVs)
		{
			return lightmappack(positions, indices, indexCount, sizeof(index_t), width, height, scale, margin, outVertices, outUVs);
		}

		template<typename index_t = short>
		static bool lightmappack(const value_t* positions, const index_t* indices, size_type indexCount, size_type indexStride, int width, int height, value_t scale, int margin, value_t* outVertices, value_t* outUVs)
		{
			auto pos = (const vec3_t*)positions;
			auto vertices = (vec3_t*)outVertices;

			for (size_type i = 0; i < indexCount; i++, indices = (index_t*)(((char*)indices) + indexStride))
				vertices[i] = pos[*indices];

			return lightmappack(outVertices, indexCount, width, height, scale, margin, outUVs);
		}

	private:
		static size_type lightmappack(std::vector<quad_t>& quad, int width, int height, value_t scale, value_t scalefactor, int margin)
		{
			value_t area = 0;

			for (auto& it : quad)
				area += it.area();

			area = std::sqrt(area) * scalefactor;

			for (auto& it : quad)
			{
				it.edge /= area;
				it.computeUV((value_t)margin / width * 0.5f);
			}

			auto border = vec2_t((value_t)margin / width, (value_t)margin / height);

			size_type processed = 0;

			quad_node_t root;

			for (auto& it : quad)
			{
				if (!root.insert(it, border))
					break;
				processed++;
			}

			return processed;
		}

		static size_type lightmappack(const std::vector<triangle_t>& triangles, int width, int height, value_t scale, value_t scalefactor, int margin, value_t* outUVs)
		{
			std::vector<triangle_t> tris = triangles;

			std::qsort(tris.data(), tris.size(), sizeof(triangle_t), [](const void* a, const void* b) -> int
			{
				auto t1 = (triangle_t*)a;
				auto t2 = (triangle_t*)b;
				auto dh = t2->h - t1->h;
				return dh != 0 ? dh : (t2->w - t1->w);
			});

			std::vector<quad_t> quad(tris.size() >> 1);

			for (size_type i = 0; i < quad.size() - tris.size() % 2; i++)
				quad[i].computeEdge(&tris[i * 2], &tris[i * 2 + 1]);

			if (tris.size() % 2 > 0)
				quad.push_back(quad_t(&(*tris.rbegin()), nullptr));

			auto processed = lightmappack(quad, width, height, scale, scalefactor, margin) * 2;

			if (tris.size() % 2 > 0)
				processed--;

			if (outUVs)
			{
				vec2_t* uv = (vec2_t*)outUVs;

				for (auto& it : tris)
				{
					(*uv++) = it.uv[0];
					(*uv++) = it.uv[1];
					(*uv++) = it.uv[2];
				}
			}

			return processed;
		}

		static size_type lightmappack(const std::vector<triangle_t>& tris, int width, int height, value_t scale, int margin, value_t* outUVs)
		{
			value_t testScale = 1.0f;
			size_type processed = lightmappack(tris, width, height, scale, testScale, margin, 0);

			if (processed > tris.size())
			{
				for (std::uint8_t j = 0; processed > tris.size() && j < 16; j++)
				{
					testScale -= 0.022;
					processed = lightmappack(tris, width, height, scale, testScale, margin, 0);
				}
			}
			else
			{
				for (std::uint8_t j = 0; processed < tris.size() && j < 16; j++)
				{
					testScale += 0.022;
					processed = lightmappack(tris, width, height, scale, testScale, margin, 0);
				}
			}

			return lightmappack(tris, width, height, scale, testScale, margin, outUVs);
		}
	};

	using uvmapper = basic_uvmapper<float>;
}

#endif