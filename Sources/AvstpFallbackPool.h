/*****************************************************************************

        AvstpFallbackPool.h

Thread pool backing AvstpWrapper's fallback path, used when avstp.dll is
not installed. The original fallback reported one thread and ran every
task inline, so mt=true did nothing without avstp.dll.

Workers are created on first use and shared by all dispatchers. They are
joined by shutdown(), which the plug-in registers with env->AtExit() so
no thread is left running inside the DLL when the host unloads it. A
later call restarts the pool.

--- Legal stuff ---

This program is free software. It comes without any warranty, to
the extent permitted by applicable law. You can redistribute it
and/or modify it under the terms of the Do What The Fuck You Want
To Public License, Version 2, as published by Sam Hocevar. See
http://sam.zoy.org/wtfpl/COPYING for more details.

*Tab=3***********************************************************************/



#if ! defined (AvstpFallbackPool_HEADER_INCLUDED)
#define	AvstpFallbackPool_HEADER_INCLUDED



/*\\\ INCLUDE FILES \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

#include "avstp.h"



/*\\\ PUBLIC \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

class AvstpFallbackPool
{
public:

	static int                     get_nbr_threads ();
	static avstp_TaskDispatcher *  create_dispatcher ();
	static void                    destroy_dispatcher (avstp_TaskDispatcher *td_ptr);
	static int                     enqueue_task (avstp_TaskDispatcher *td_ptr, avstp_TaskPtr task_ptr, void *user_data_ptr);
	static int                     wait_completion (avstp_TaskDispatcher *td_ptr);
	static void                    shutdown ();

private:

	               AvstpFallbackPool ()                                       = delete;
	               AvstpFallbackPool (const AvstpFallbackPool &other)         = delete;
	               AvstpFallbackPool (AvstpFallbackPool &&other)              = delete;
	AvstpFallbackPool &
	               operator = (const AvstpFallbackPool &other)                = delete;
	AvstpFallbackPool &
	               operator = (AvstpFallbackPool &&other)                     = delete;

};	// class AvstpFallbackPool



#endif	// AvstpFallbackPool_HEADER_INCLUDED



/*\\\ EOF \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
